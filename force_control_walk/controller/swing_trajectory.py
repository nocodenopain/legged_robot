import numpy as np

def SwingTrajectoryBezier(pf_init, pf_final, phase, swingtime, h):
    # 计算位置、速度和加速度
    pout = cubicBezier(pf_init, pf_final, phase)
    p_v = cubicBezier_v(pf_init, pf_final, phase) / swingtime
    p_a = cubicBezier_a(pf_init, pf_final, phase) / (swingtime * swingtime)

    # 根据 phase 值判断是否在上升阶段
    if phase < 0.5:
        zp = cubicBezier(pf_init[2], pf_init[2] + h, phase * 2)
        zv = cubicBezier_v(pf_init[2], pf_init[2] + h, phase * 2) * 2 / swingtime
        za = cubicBezier_a(pf_init[2], pf_init[2] + h, phase * 2) * 4 / (swingtime * swingtime)
    else:
        zp = cubicBezier(pf_init[2] + h, pf_final[2], phase * 2 - 1)
        zv = cubicBezier_v(pf_init[2] + h, pf_final[2], phase * 2 - 1) * 2 / swingtime
        za = cubicBezier_a(pf_init[2] + h, pf_final[2], phase * 2 - 1) * 4 / (swingtime * swingtime)

    # 更新 z 轴的值
    pout[2] = zp
    p_v[2] = zv
    p_a[2] = za

    return pout, p_v, p_a

def cubicBezier(p0, pf, t):
    return p0 + (t**3 + 3 * t**2 * (1 - t)) * (pf - p0)

def cubicBezier_v(p0, pf, t):
    return 6 * t * (1 - t) * (pf - p0)

def cubicBezier_a(p0, pf, t):
    return (6 - 12 * t) * (pf - p0)
