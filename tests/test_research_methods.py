"""研究方法的独立数值检查：坐标、面积、配对、时间窗口及状态转移。"""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd
from scipy.linalg import expm
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from 构建沙洲观测与变化表 import normalize_cay_mask, select_analysis_observations, build_transition
from 分析沙洲面积轨迹 import track_shape_metrics
from 对齐流场波浪与沙洲变化 import decompose_along_cross, aggregate_source, bh_fdr
from 拟合Sentinel2三状态连续时间模型 import theta_to_q, model_rows, negative_log_likelihood, collapse_likelihood_terms, make_pairs, transition_probabilities
from 提取沙洲颜色组成 import calibrated_colour_pixels, vegetation_pixels
from 分析季节相位质心往复 import summarize_pairs


def observation(sensor="sentinel2", frame="S2", image="a", date="2020-01-01", **extra):
    return dict(sensor=sensor, reference_frame_id=frame, image_id=image, date=pd.Timestamp(date),
                sand_cay_id="TEST_cay_001", reef_id="TEST", quality_grade="A", **extra)


class ResearchMethodsTests(unittest.TestCase):
    def test_mobility_grouping_uses_accepted_cover_not_historical_stage(self):
        from 分析裸沙质心轨迹周期与突发 import build_steps
        observations=pd.DataFrame([observation(image='start',date='2020-01-01'),
                                   observation(image='end',date='2020-03-01')])
        interval=dict(transition_id='test',sensor='sentinel2',sand_cay_id='TEST_cay_001',reef_id='TEST',
                      reference_frame_id='S2',time_t='2020-01-01',time_t1='2020-03-01',
                      time_interval_days=60,centroid_shift_m=2.,area_t_m2=100.)
        labels=observations.copy()
        labels['analysis_vegetation_state']=['sparse','none']
        labels['development_stage']='stage3_stable_exposed'
        with patch('分析范围.load_excluded_reefs',return_value=set()), patch('pandas.read_csv',side_effect=[pd.DataFrame([interval]),labels]):
            steps=build_steps(observations,grouping='cover')
        self.assertEqual(steps.stratum.tolist(),['vegetated'])
        self.assertEqual(steps.cover_state_t.tolist(),['sparse'])
        self.assertAlmostEqual(steps.speed_norm.iloc[0],2/(100/np.pi)**.5/(60/365.2425))
        # 缺少有效末期观测时，该时段必须排除，不能仅凭期初标签进入比较。
        with patch('分析范围.load_excluded_reefs',return_value=set()), patch('pandas.read_csv',side_effect=[pd.DataFrame([interval]),labels]):
            steps=build_steps(observations.iloc[:1],grouping='cover')
        self.assertTrue(steps.empty)

    def test_pixel_rules_share_the_same_mask_and_green_fraction(self):
        image = np.full((20, 20, 3), 220, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:14, 4:14] = 1
        image[4:10, 4:10] = [20, 160, 20]
        data = calibrated_colour_pixels(image, mask)
        self.assertAlmostEqual(float(data["vegetation"].mean()), .36)
        self.assertEqual(int(vegetation_pixels(image, mask).sum()), 36)

    def test_transition_probabilities_match_scipy_in_regular_and_degenerate_cases(self):
        times = np.array([0., .1, .5, 1., 3.])
        regular = theta_to_q(np.log([.2,.1,.3,.1,.2,.4]))
        defective = np.array([[-1.,1.,0.],[0.,-1.,1.],[0.,0.,0.]])
        for q in [regular, defective]:
            actual = transition_probabilities(q, times)
            expected = np.array([expm(q*t) for t in times])
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_main_component_has_known_area_and_centroid(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 3:7] = 1
        mask[9, 9] = 1
        clean = normalize_cay_mask(mask)
        self.assertEqual(int(clean.sum()), 16)
        y, x = np.nonzero(clean)
        self.assertEqual((y.mean(), x.mean()), (3.5, 4.5))

    def test_observation_selection_is_source_specific_and_not_area_selected(self):
        rows = [observation(image="a", pixel_count=1), observation(image="z", pixel_count=99),
                observation(sensor="google_earth", frame="GE", image="g", pixel_count=50)]
        frame = pd.DataFrame(rows)
        selected = select_analysis_observations(frame)
        self.assertEqual(set(selected.image_id), {"a", "g"})
        bad = observation(image="bad", date="2020-02-01", pixel_count=100)
        bad["quality_grade"] = "C"
        selected = select_analysis_observations(pd.concat([frame, pd.DataFrame([bad])], ignore_index=True))
        self.assertNotIn("bad", selected.image_id.tolist())

    def test_translated_square_preserves_area_balance_and_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            records = []
            for i, left in enumerate([2, 3]):
                mask = np.zeros((10, 10), dtype=np.uint8)
                mask[2:6, left:left + 4] = 255
                path = Path(directory) / f"沙洲_{i}.png"
                ok, encoded = cv2.imencode(".png", mask)
                self.assertTrue(ok)
                encoded.tofile(path)
                records.append(pd.Series(observation(image=str(i), date=f"2020-01-0{i+1}",
                    pixel_size_m=10., mask_path=str(path), sand_cay_area_m2=1600.,
                    sand_cay_centroid_x=left + 1.5, sand_cay_centroid_y=3.5)))
            result = build_transition(records[0], records[1], {})
            self.assertEqual(result["area_change_m2"], 0)
            self.assertEqual(result["erosion_area_m2"], 400)
            self.assertEqual(result["deposition_area_m2"], 400)
            self.assertEqual(result["gross_boundary_change_m2"], 800)
            self.assertEqual(result["centroid_east_shift_m"], 10)
            self.assertEqual(result["centroid_north_shift_m"], 0)

    def test_centroid_envelope_is_translation_invariant(self):
        frame = pd.DataFrame(dict(date=pd.date_range("2020-01-01", periods=3),
            sand_cay_area_m2=[100*np.pi]*3, sand_cay_perimeter_m=[20]*3,
            major_axis_length_m=[12]*3, minor_axis_length_m=[8]*3,
            sand_cay_major_axis_angle=[0]*3, pixel_size_m=[1]*3,
            sand_cay_centroid_x=[0,10,0], sand_cay_centroid_y=[0,0,0]))
        first = track_shape_metrics(frame)
        frame.sand_cay_centroid_x += 1000
        frame.sand_cay_centroid_y += 2000
        second = track_shape_metrics(frame)
        self.assertAlmostEqual(first["centroid_envelope_normalized"], 2/3)
        self.assertAlmostEqual(first["centroid_path_normalized"], 2)
        self.assertAlmostEqual(first["centroid_net_normalized"], 0)
        self.assertAlmostEqual(first["centroid_envelope_normalized"], second["centroid_envelope_normalized"])

    def test_vector_projection_uses_its_own_sensor_frame(self):
        obs = pd.DataFrame([observation(sensor="google_earth", frame="GE", image="g", sand_cay_major_axis_angle=0),
                            observation(frame="S2", image="s", sand_cay_major_axis_angle=90)])
        rows = []
        for sensor, frame, key in [("google_earth", "GE", "g"), ("sentinel2", "S2", "s")]:
            rows.append(dict(sensor=sensor, reference_frame_id=frame, sand_cay_id="TEST_cay_001",
                transition_id=key, time_t="2020-01-01", time_interval_days=365.2425, area_t_m2=100.,
                current_u_mean=1., current_v_mean=0., wave_vector_u_mean=1., wave_vector_v_mean=0.,
                wind_u_mean=1., wind_v_mean=0., centroid_east_shift_m=20., centroid_north_shift_m=0.,
                major_axis_change_m=0., minor_axis_change_m=0.))
        result = decompose_along_cross(pd.DataFrame(rows), obs).set_index("sensor")
        self.assertAlmostEqual(result.loc["sentinel2", "current_along_mean"], 1)
        self.assertAlmostEqual(result.loc["sentinel2", "centroid_along_shift_normalized_per_year"], 2)
        self.assertAlmostEqual(result.loc["google_earth", "current_along_mean"], 0)
        self.assertAlmostEqual(result.loc["google_earth", "current_cross_mean"], 1)

    def test_environment_window_excludes_start_and_rejects_missing_days(self):
        interval = pd.DataFrame([dict(transition_id="t", reef_id="TEST", sand_cay_id="TEST_cay_001",
                                     time_t="2020-01-01", time_t1="2020-01-03")])
        daily = pd.DataFrame(dict(source=["era5"]*3, reef_id=["TEST"]*3,
                                 time=pd.date_range("2020-01-01", periods=3), u10=[99.,2.,6.], v10=[0.]*3))
        result, _ = aggregate_source(interval, daily, "era5", ("u10", "v10"))
        self.assertEqual(result.era5_complete.iloc[0], 1)
        self.assertEqual(result.wind_u_mean.iloc[0], 4)
        incomplete, _ = aggregate_source(interval, daily.drop(index=1), "era5", ("u10", "v10"))
        self.assertEqual(incomplete.era5_complete.iloc[0], 0)
        self.assertNotIn("wind_u_mean", incomplete.columns)

    def test_fdr_matches_known_bh_values(self):
        result = bh_fdr(pd.Series([.01,.03,.04,.20,np.nan]))
        np.testing.assert_allclose(result.iloc[:4], [.04,.04*4/3,.04*4/3,.20])
        self.assertTrue(pd.isna(result.iloc[4]))

    def test_ctmc_generator_probability_and_mean_holding_time(self):
        q = theta_to_q(np.log([.2,.1,.3,.1,.2,.4]))
        np.testing.assert_allclose(q.sum(axis=1), 0, atol=1e-15)
        p = expm(q * .5)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
        self.assertGreaterEqual(float(p.min()), 0)
        rows = model_rows(q, "test", 10, 3, 1.)
        means = [r["mean_holding_years"] for r in rows if r["record_type"] == "state_summary"]
        np.testing.assert_allclose(means, [1/.3,1/.4,1/.6])

    def test_ctmc_collapsed_likelihood_matches_explicit_likelihood(self):
        theta = np.log([.2,.1,.3,.1,.2,.4])
        pairs = pd.DataFrame(dict(from_index=[0,0,1,2],to_index=[1,1,0,2],interval_years=[.5,.5,1.,.25]))
        q = theta_to_q(theta)
        reference = -sum(np.log(expm(q*r.interval_years)[r.from_index,r.to_index]) for r in pairs.itertuples())
        self.assertAlmostEqual(negative_log_likelihood(theta,collapse_likelihood_terms(pairs)), reference)

    def test_ctmc_pairs_do_not_cross_source_or_frame(self):
        rows = []
        for sensor, frame in [("sentinel2", "S2"),("google_earth","GE"),("sentinel2","OTHER_FRAME")]:
            for i, state in enumerate(["low_cover","dominant_cover"]):
                rows.append(observation(sensor=sensor,frame=frame,image=f"{frame}_{i}",date=["2020-01-01","2020-02-15"][i],
                                        analysis_cover_state_3=state,cover_state_status_3="confirmed"))
        pairs = make_pairs(pd.DataFrame(rows),min_days=30,max_days=365,confirmed_only=False)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(pairs.interval_days.eq(45).all())


class 文稿身份保留检验(unittest.TestCase):
    def test_math_subscripts_do_not_rewrite_data_identifiers(self):
        from 生成中文论文 import clean_inline
        identifiers = "GBR_101 MEDF_015_cay_001 sand_cay_id time_interval_days"
        self.assertEqual(clean_inline(identifiers), identifiers)
        self.assertEqual(clean_inline("Q_ii P_ij A_i x_1"), "Qᵢᵢ Pᵢⱼ Aᵢ x₁")


if __name__ == "__main__":
    unittest.main()
