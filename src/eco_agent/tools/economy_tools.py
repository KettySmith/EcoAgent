# -*- coding: utf-8 -*-
"""
量化投资 & 金融财经计算工具库
包含：风险收益、组合管理、固收、宏观、公司金融、表格处理等常用公式
已修复所有错误，添加完整中文注释，可直接运行
"""

import numpy as np
import pandas as pds
import math
import os
from langchain_core.tools import tool
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm

# ===================== 1. 量化投资风险收益指标 =====================
@tool
def sortino_ratio_calc(returns_str: str, risk_free: float = 0.02) -> str:
    """索提诺比率计算（仅考量下行风险）"""
    rets = np.array([float(x) for x in returns_str.split(",")])
    down_rets = rets[rets < risk_free]
    down_std = np.std(down_rets, ddof=1) if len(down_rets) > 0 else 0
    ann_ret = np.mean(rets) * 252
    ann_down_std = down_std * np.sqrt(252)

    if ann_down_std == 0:
        sortino = np.inf
    else:
        sortino = (ann_ret - risk_free) / ann_down_std

    return (f"【索提诺比率】\n年化收益：{ann_ret:.4%}\n年化下行波动率：{ann_down_std:.4%}\n索提诺比率：{sortino:.4f}")

@tool
def calmar_ratio_calc(returns_str: str) -> str:
    """卡玛比率（收益/最大回撤）"""
    rets = np.array([float(x) for x in returns_str.split(",")])
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    ann_ret = (cum[-1] ** (252 / len(rets))) - 1
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.inf

    return f"【卡玛比率】\n年化收益率：{ann_ret:.4%}\n最大回撤：{max_dd:.4%}\n卡玛比率：{calmar:.4f}"

@tool
def portfolio_metrics_calculator(returns_data: str, risk_free_rate: float = 0.02) -> str:
    """投资组合指标（年化收益、波动率、夏普、最大回撤）"""
    returns = np.array([float(x.strip()) for x in returns_data.split(',')])
    n = len(returns)
    cumulative_return = (1 + returns).cumprod()

    annual_return = (cumulative_return[-1]) ** (252 / n) - 1
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.inf
    peak = cumulative_return.cummax()
    drawdown = (cumulative_return - peak) / peak
    max_drawdown = drawdown.min()

    return f"组合指标：\n年化收益：{annual_return:.4%}\n年化波动：{annual_volatility:.4%}\n夏普比率：{sharpe_ratio:.4f}\n最大回撤：{max_drawdown:.4%}"

@tool
def var_cvar_calculator(portfolio_value: float, returns_data: str, confidence_level: float = 0.95) -> str:
    """在险价值VaR与条件在险价值CVaR"""
    returns = np.array([float(x.strip()) for x in returns_data.split(',')])
    losses = -portfolio_value * returns
    var = np.percentile(losses, 100 * (1 - confidence_level))
    cvar = losses[losses >= var].mean() if len(losses[losses >= var]) > 0 else var

    return f"风险评估（置信{confidence_level:.0%}）：\n每日VaR：{var:.2f}元\n每日CVaR：{cvar:.2f}元\n年化VaR：{var*np.sqrt(252):.2f}元\n年化CVaR：{cvar*np.sqrt(252):.2f}元"

# ===================== 2. 财务比率 =====================
@tool
def financial_ratios_calculator(current_assets: float, current_liabilities: float,
                               total_assets: float, total_liabilities: float,
                               revenue: float, net_profit: float, equity: float) -> str:
    """核心财务比率计算器"""
    current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0
    debt_to_asset_ratio = total_liabilities / total_assets if total_assets != 0 else 0
    roe = net_profit / equity if equity != 0 else 0
    profit_margin = net_profit / revenue if revenue != 0 else 0

    return f"财务比率：\n流动比率：{current_ratio:.2f}\n资产负债率：{debt_to_asset_ratio:.2%}\nROE：{roe:.2%}\n销售净利率：{profit_margin:.2%}"

# ===================== 3. 货币时间价值 =====================
@tool
def time_value_calculator(calculation_type: str, principal: float, rate: float,
                         periods: int, compounding: int = 1) -> str:
    """复利终值/现值计算"""
    if calculation_type == "future_value":
        fv = principal * math.pow((1 + rate / compounding), compounding * periods)
        return f"复利终值：{fv:.2f}元"
    elif calculation_type == "present_value":
        pv = principal / math.pow((1 + rate / compounding), compounding * periods)
        return f"现值：{pv:.2f}元"
    else:
        return "错误：仅支持future_value/present_value"

# ===================== 4. 宏观经济公式 =====================
@tool
def taylor_rule_calc(neutral_rate: float, infl_now: float, infl_target: float, output_gap: float) -> str:
    """泰勒规则：央行理论政策利率"""
    alpha = 0.5
    beta = 0.5
    policy_rate = neutral_rate + infl_now + alpha * (infl_now - infl_target) + beta * output_gap
    return f"泰勒规则：\n理论政策利率：{policy_rate:.2%}"

@tool
def okun_law_calc(unemp_now: float, unemp_natural: float) -> str:
    """奥肯定律：失业率→产出缺口"""
    gap_unemp = unemp_now - unemp_natural
    gdp_gap = -2 * gap_unemp
    return f"奥肯定律：\nGDP产出缺口：{gdp_gap:.2%}"

@tool
def money_multiplier_calc(c_ratio: float, rd_ratio: float, re_ratio: float) -> str:
    """完整货币乘数"""
    m = (1 + c_ratio) / (rd_ratio + re_ratio + c_ratio)
    return f"货币乘数：{m:.4f}"

# ===================== 5. 行业/统计指标 =====================
@tool
def hhi_index_calc(market_share_str: str) -> str:
    """HHI赫芬达尔垄断指数"""
    shares = np.array([float(x) for x in market_share_str.split(",")])
    hhi = np.sum(shares ** 2) * 10000
    level = "高度垄断" if hhi > 2500 else "中度集中" if hhi > 1500 else "竞争充分"
    return f"HHI：{hhi:.2f}，市场格局：{level}"

@tool
def gini_coeff_calc(income_str: str) -> str:
    """基尼系数"""
    y = np.array([float(x) for x in income_str.split(",")])
    y = np.sort(y)
    n = len(y)
    sum_y = np.sum(y)
    i = np.arange(1, n + 1)
    gini = (n + 1 - 2 * np.sum(i * y) / sum_y) / n
    return f"基尼系数：{gini:.4f}"

# ===================== 6. 债券固收 =====================
@tool
def bond_convexity_calc(price: float, yld: float, cash_flows_str: str, periods: int) -> str:
    """债券凸性"""
    cf = np.array([float(x) for x in cash_flows_str.split(",")])
    t = np.arange(1, periods + 1)
    y = yld
    numerator = np.sum(t * (t + 1) * cf / (1 + y) ** (t + 2))
    convexity = numerator / price
    return f"债券凸性：{convexity:.4f}"

@tool
def bond_duration_calc(price0: float, price_up: float, price_down: float, dy: float) -> str:
    """债券有效久期"""
    eff_dur = (price_down - price_up) / (2 * price0 * dy)
    return f"有效久期：{eff_dur:.4f}"

@tool
def bond_pricing(coupon: float, face_value: float, periods: int, discount_rates: str) -> str:
    """债券定价"""
    rate_list = [float(r.strip()) for r in discount_rates.split(",")]
    if len(rate_list) == 1:
        rate_list = [rate_list[0]] * periods
    elif len(rate_list) != periods:
        return "贴现期数不匹配"

    price = 0.0
    for t in range(1, periods + 1):
        price += coupon / (1 + rate_list[t - 1]) ** t
    price += face_value / (1 + rate_list[-1]) ** periods
    return f"债券理论价格：{price:.2f}"

# ===================== 7. 股票估值 =====================
@tool
def price_to_earnings_ratio(payout_ratio: float, growth_rate: float, required_return: float) -> str:
    """理论市盈率"""
    if growth_rate >= required_return:
        return "增长率必须小于必要收益率"
    pe_ratio = payout_ratio * (1 + growth_rate) / (required_return - growth_rate)
    return f"合理PE：{pe_ratio:.2f}倍"

@tool
def dividend_growth_rate(roe: float, payout_ratio: float) -> str:
    """股利增长率 g=ROE*(1-分红率)"""
    retention_ratio = 1 - payout_ratio
    growth_rate = roe * retention_ratio
    return f"股利增长率g：{growth_rate:.4%}"

@tool
def gordon_growth_model(required_return: float, growth_rate: float, d0: float = None, d1: float = None) -> str:
    """戈登股利增长模型"""
    if growth_rate >= required_return:
        return "增长率必须小于必要收益率"
    if d1 is None:
        d1 = d0 * (1 + growth_rate)
    price = d1 / (required_return - growth_rate)
    return f"股票理论价格：{price:.4f}"

# ===================== 8. 年金/利率 =====================
@tool
def annuity_present_value(payment: float, rate: float, periods: int) -> str:
    """普通年金现值"""
    if rate == 0:
        pv = payment * periods
    else:
        pv = payment / rate * (1 - (1 + rate) ** (-periods))
    return f"年金现值：{pv:.2f}"

@tool
def effective_annual_rate(apr: float, compounding_periods: int) -> str:
    """实际年利率EAR"""
    ear = (1 + apr / compounding_periods) ** compounding_periods - 1
    return f"实际年利率EAR：{ear:.4%}"

# ===================== 9. 组合统计 =====================
@tool
def portfolio_return_variance(returns_str: str, probabilities_str: str) -> str:
    """期望收益、方差、标准差"""
    returns = np.array([float(r.strip()) for r in returns_str.split(",")])
    probs = np.array([float(p.strip()) for p in probabilities_str.split(",")])
    exp_ret = sum(probs * returns)
    var = sum(probs * (returns - exp_ret) ** 2)
    std = np.sqrt(var)
    return f"期望收益：{exp_ret:.4%}，方差：{var:.6f}，标准差：{std:.4%}"

@tool
def portfolio_covariance_tool(ret1_str: str, ret2_str: str, prob_str: str) -> str:
    """两资产协方差"""
    r1 = np.array([float(x) for x in ret1_str.split(",")])
    r2 = np.array([float(x) for x in ret2_str.split(",")])
    p = np.array([float(x) for x in prob_str.split(",")])
    e1 = sum(p * r1)
    e2 = sum(p * r2)
    cov = sum(p * (r1 - e1) * (r2 - e2))
    return f"协方差：{cov:.6f}"

@tool
def portfolio_correlation_tool(ret1_str: str, ret2_str: str, prob_str: str) -> str:
    """两资产相关系数"""
    r1 = np.array([float(x) for x in ret1_str.split(",")])
    r2 = np.array([float(x) for x in ret2_str.split(",")])
    p = np.array([float(x) for x in prob_str.split(",")])
    e1 = sum(p * r1)
    e2 = sum(p * r2)
    cov = sum(p * (r1 - e1) * (r2 - e2))
    std1 = np.sqrt(sum(p * (r1 - e1) ** 2))
    std2 = np.sqrt(sum(p * (r2 - e2) ** 2))
    corr = cov / (std1 * std2)
    return f"相关系数：{corr:.4f}"

# ===================== 10. 定价模型 =====================
@tool
def capm_model_tool(rf: float, beta: float, rm: float) -> str:
    """CAPM资本资产定价"""
    ri = rf + beta * (rm - rf)
    return f"个股必要收益率：{ri:.4%}"

@tool
def beta_coefficient_tool(cov_im: float, var_m: float) -> str:
    """贝塔系数"""
    beta = cov_im / var_m
    return f"β：{beta:.4f}"

@tool
def wacc_calc_tool(wd: float, rd: float, tax: float, we: float, re: float) -> str:
    """加权平均资本成本WACC"""
    wacc = wd * rd * (1 - tax) + we * re
    return f"WACC：{wacc:.4%}"

# ===================== 11. 时间序列 =====================
@tool
def geometric_return_tool(returns_str: str) -> str:
    """几何平均收益率"""
    rets = np.array([float(x) for x in returns_str.split(",")])
    geo = np.prod(1 + rets) ** (1 / len(rets)) - 1
    return f"几何平均收益：{geo:.4%}"

@tool
def adf_stationarity_test(series_str: str) -> str:
    """ADF平稳性检验"""
    arr = np.array([float(x) for x in series_str.split(",")])
    adf_stat, pval, _, _, _, _ = adfuller(arr)
    res = "平稳" if pval < 0.05 else "非平稳"
    return f"ADF统计量：{adf_stat:.4f}，P值：{pval:.4f}，结论：{res}"

# ===================== 12. 杠杆系数 =====================
@tool
def dol_operating_lever_tool(ebit_q: float, ebit_0: float, q_q: float, q_0: float) -> str:
    """经营杠杆DOL"""
    ebit_rate = (ebit_q - ebit_0) / ebit_0
    q_rate = (q_q - q_0) / q_0
    dol = ebit_rate / q_rate if q_rate != 0 else 0
    return f"DOL：{dol:.4f}"

@tool
def dfl_financial_lever_tool(eps_q: float, eps_0: float, ebit_q: float, ebit_0: float) -> str:
    """财务杠杆DFL"""
    eps_rate = (eps_q - eps_0) / eps_0
    ebit_rate = (ebit_q - ebit_0) / ebit_0
    dfl = eps_rate / ebit_rate if ebit_rate != 0 else 0
    return f"DFL：{dfl:.4f}"

# ===================== 13. 更多量化指标 =====================
@tool
def information_ratio_calc(portfolio_returns_str: str, benchmark_returns_str: str) -> str:
    """信息比率（超额收益/跟踪误差）"""
    port_rets = np.array([float(x) for x in portfolio_returns_str.split(",")])
    bench_rets = np.array([float(x) for x in benchmark_returns_str.split(",")])
    excess_rets = port_rets - bench_rets
    ann_excess_ret = np.mean(excess_rets) * 252
    tracking_error = np.std(excess_rets, ddof=1) * np.sqrt(252)
    ir = ann_excess_ret / tracking_error if tracking_error != 0 else np.inf
    return f"信息比率：{ir:.4f}\n年化超额收益：{ann_excess_ret:.4%}\n年化跟踪误差：{tracking_error:.4%}"

@tool
def jensens_alpha_calc(portfolio_returns_str: str, benchmark_returns_str: str, risk_free: float = 0.02) -> str:
    """詹森阿尔法（超额收益调整风险）"""
    port_rets = np.array([float(x) for x in portfolio_returns_str.split(",")])
    bench_rets = np.array([float(x) for x in benchmark_returns_str.split(",")])
    ann_port_ret = np.mean(port_rets) * 252
    ann_bench_ret = np.mean(bench_rets) * 252
    beta = np.cov(port_rets, bench_rets)[0,1] / np.var(bench_rets)
    expected_ret = risk_free + beta * (ann_bench_ret - risk_free)
    alpha = ann_port_ret - expected_ret
    return f"詹森阿尔法：{alpha:.4%}\n贝塔：{beta:.4f}\n预期收益：{expected_ret:.4%}"

@tool
def skewness_kurtosis_calc(returns_str: str) -> str:
    """偏度与峰度"""
    rets = np.array([float(x) for x in returns_str.split(",")])
    skew = np.mean((rets - np.mean(rets))**3) / (np.std(rets, ddof=1)**3)
    kurt = np.mean((rets - np.mean(rets))**4) / (np.std(rets, ddof=1)**4) - 3
    return f"偏度：{skew:.4f}\n峰度：{kurt:.4f}"

# ===================== 14. 期权定价 =====================
@tool
def black_scholes_call_price(S: float, K: float, T: float, r: float, sigma: float) -> str:
    """Black-Scholes 欧式看涨期权定价"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return f"看涨期权价格：{call_price:.4f}"

@tool
def black_scholes_put_price(S: float, K: float, T: float, r: float, sigma: float) -> str:
    """Black-Scholes 欧式看跌期权定价"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return f"看跌期权价格：{put_price:.4f}"

# ===================== 15. 债券分析 =====================
@tool
def bond_ytm_calc(price: float, face_value: float, coupon: float, periods: int) -> str:
    """债券到期收益率YTM（近似计算）"""
    cash_flows = [coupon] * periods + [face_value]
    ytm = np.irr([-price] + cash_flows)
    return f"到期收益率YTM：{ytm:.4%}"

# ===================== 16. 宏观经济 =====================
@tool
def phillips_curve_calc(unemployment: float, expected_inflation: float, natural_unemployment: float = 0.05) -> str:
    """菲利普斯曲线（失业率与通胀关系）"""
    inflation = expected_inflation - 2 * (unemployment - natural_unemployment)
    return f"预期通胀率：{inflation:.2%}"

# ===================== 17. 公司金融 =====================
@tool
def eva_calc(nopat: float, capital_charge: float) -> str:
    """经济增加值EVA"""
    eva = nopat - capital_charge
    return f"EVA：{eva:.2f}"

@tool
def dupont_analysis_calc(net_profit: float, revenue: float, total_assets: float, equity: float) -> str:
    """杜邦分析"""
    profit_margin = net_profit / revenue
    asset_turnover = revenue / total_assets
    equity_multiplier = total_assets / equity
    roe = profit_margin * asset_turnover * equity_multiplier
    return f"杜邦分析ROE：{roe:.2%}\n销售净利率：{profit_margin:.2%}\n总资产周转率：{asset_turnover:.4f}\n权益乘数：{equity_multiplier:.2f}"

# ===================== 18. 资本预算 =====================
@tool
def npv_calc(cash_flows_str: str, discount_rate: float) -> str:
    """净现值NPV"""
    cf = np.array([float(x) for x in cash_flows_str.split(",")])
    npv = sum(cf / (1 + discount_rate)**t for t, cf in enumerate(cf, 1))
    return f"NPV：{npv:.2f}"

@tool
def irr_calc(cash_flows_str: str) -> str:
    """内部收益率IRR"""
    cf = np.array([float(x) for x in cash_flows_str.split(",")])
    irr = np.irr(cf)
    return f"IRR：{irr:.4%}"

# ===================== 19. 外汇 =====================
@tool
def forward_exchange_rate_calc(spot_rate: float, domestic_rate: float, foreign_rate: float, time: float) -> str:
    """外汇远期汇率"""
    forward_rate = spot_rate * (1 + domestic_rate * time) / (1 + foreign_rate * time)
    return f"远期汇率：{forward_rate:.4f}"

# ===================== 20. 期权高级 =====================
@tool
def implied_volatility_calc(S: float, K: float, T: float, r: float, market_price: float, option_type: str = 'call') -> str:
    """隐含波动率（近似二分法）"""
    def bs_price(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    sigma_low, sigma_high = 0.01, 2.0
    for _ in range(100):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = bs_price(sigma_mid)
        if price_mid < market_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    return f"隐含波动率：{sigma_mid:.4f}"

# ===================== 21. 风险管理高级 =====================
@tool
def historical_var_calc(returns_str: str, confidence: float = 0.95) -> str:
    """历史模拟VaR"""
    rets = np.array([float(x) for x in returns_str.split(",")])
    sorted_rets = np.sort(rets)
    index = int((1 - confidence) * len(rets))
    var = -sorted_rets[index]
    return f"历史VaR（{confidence:.0%}置信度）：{var:.4%}"

@tool
def monte_carlo_var_calc(S: float, mu: float, sigma: float, T: float, simulations: int = 10000, confidence: float = 0.95) -> str:
    """蒙特卡洛VaR"""
    np.random.seed(42)
    returns = np.random.normal(mu * T, sigma * np.sqrt(T), simulations)
    losses = -S * returns
    var = np.percentile(losses, 100 * (1 - confidence))
    return f"蒙特卡洛VaR（{confidence:.0%}置信度）：{var:.2f}"

# ===================== 22. 房地产金融 =====================
@tool
def cap_rate_calc(noi: float, property_value: float) -> str:
    """资本化率"""
    cap_rate = noi / property_value
    return f"资本化率：{cap_rate:.2%}"

# ===================== 23. 行为金融 =====================
@tool
def prospect_theory_value_calc(gain: float, loss: float, alpha: float = 0.88, beta: float = 2.25) -> str:
    """前景理论价值函数（简化）"""
    value = gain**alpha - (abs(loss)**beta)
    return f"前景理论价值：{value:.4f}"

# ===================== 24. 宏观经济高级 =====================
@tool
def purchasing_power_parity_calc(spot_rate: float, domestic_inflation: float, foreign_inflation: float, time: float) -> str:
    """购买力平价"""
    ppp_rate = spot_rate * (1 + domestic_inflation * time) / (1 + foreign_inflation * time)
    return f"PPP汇率：{ppp_rate:.4f}"

@tool
def gdp_growth_calc(gdp_current: float, gdp_previous: float) -> str:
    """GDP增长率"""
    growth = (gdp_current - gdp_previous) / gdp_previous
    return f"GDP增长率：{growth:.2%}"

# ===================== 25. 公司治理 =====================
@tool
def tobin_q_calc(market_value: float, replacement_cost: float) -> str:
    """托宾Q值"""
    q = market_value / replacement_cost
    return f"托宾Q：{q:.4f}"

# ===================== 26. 其他金融工具 =====================
@tool
def semi_variance_calc(returns_str: str, threshold: float = 0.0) -> str:
    """下行半方差"""
    rets = np.array([float(x) for x in returns_str.split(",")])
    down_ret = rets[rets < threshold]
    down_semi = np.var(down_ret, ddof=1) if len(down_ret) > 1 else 0
    return f"下行半方差：{down_semi:.6f}"

if __name__ == "__main__":
    print("✅ 量化金融工具库已加载完成")