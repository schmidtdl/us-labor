import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def calculate_release_ratio(
        df: pd.DataFrame,
        period: int,
) -> float:
    household_survey = df['employed'].iloc[-period:]
    establishment_survey = df['nonfarm payrolls'].iloc[-period:]
    return (establishment_survey / household_survey).mean()


def seasonal_adjust(
        df: pd.DataFrame,
        period: int = 12,
        seasonality_threshold: float = 0.1,
) -> pd.DataFrame:
    sa = {}
    for col in df.columns:
        series = df[col].dropna()
        stl = STL(series, period=period, robust=True)
        res = stl.fit()

        seasonal_variance_ratio = res.seasonal.var() / series.var()
        if seasonal_variance_ratio >= seasonality_threshold:
            sa[col] = df[col] - res.seasonal
        else:
            sa[col] = df[col]
    return pd.DataFrame(sa, index=df.index)


def calculate_sahm_rule(
        df: pd.DataFrame,
        threshold = 0.5,
) -> pd.DataFrame:
    ur = df['unemployment rate']
    ma3 = ur.rolling(3).mean()
    rolling_min = ma3.shift(1).rolling(12).min()
    indicator = ma3 - rolling_min

    return pd.DataFrame({
        'unemployment_rate': ur,
        'ma3': ma3.round(3),
        'rolling_min': rolling_min.round(3),
        'sahm_indicator': indicator.round(3),
        'recession_signal': indicator >= threshold,
    })


def extrapolate_sahm_trigger(
        df: pd.DataFrame,
        threshold: float = 0.5,
) -> float:
    rolling_min = df['rolling_min'].iloc[-1]
    prior_two_months = df['unemployment_rate'].iloc[-3:-1].sum()
    return (rolling_min + threshold) * 3 - prior_two_months


def return_beveridge_curve(
        df: pd.DataFrame,
        include_covid: bool,
) -> pd.DataFrame:
    beveridge = df[['job openings', 'unemployment rate']]
    if not include_covid:
        beveridge = beveridge.loc[
            (beveridge.index < '2020-03-01') |
            (beveridge.index > '2021-12-31')
        ]
    return beveridge.dropna()


def calculate_trend_growth(
        df: pd.DataFrame,
        span: int = 12,
) -> pd.DataFrame:
    nfp_change = df['nonfarm payrolls'].diff()
    nfp_ema = nfp_change.ewm(span=span, adjust=False).mean()
    return pd.DataFrame({
        'nfp': nfp_ema,
    }, index=df.index)


class UnemploymentCalculator:
    def __init__(self, df: pd.DataFrame):
        last = df.iloc[-1]
        self.last_date = last.name
        self.last_population = last['population']
        self.last_employment = last['employed']
        self.last_participation = last['participation rate'] / 100
        self.last_unemployment_rate = last['unemployment rate'] / 100
        self.birth_death_model = 0.127

    def calculate_breakeven_payroll(
            self,
            months: int,
            source_ratio: float,
            net_migration_growth_estimate: float,
            target_unemployment_rate: float,
            target_participation_rate: float,
    ) -> float:
        annual_population_growth = net_migration_growth_estimate + self.birth_death_model
        projected_population = self.last_population * (1 + annual_population_growth / 100 / 12 * months)
        target_employment = projected_population * target_participation_rate * (1 - target_unemployment_rate)
        return (target_employment - self.last_employment) / months * source_ratio

    def build_breakeven_table(
            self,
            months: int,
            source_ratio: float,
            net_migration_growth_estimate: float,
            step: float = 0.001,
    ) -> pd.DataFrame:
        annual_population_growth = net_migration_growth_estimate + self.birth_death_model
        ur_range = np.arange(self.last_unemployment_rate - 5 * step, self.last_unemployment_rate + 6 * step, step)
        pr_range = np.arange(self.last_participation - 5 * step, self.last_participation + 6 * step, step)
        projected_population = self.last_population * (1 + annual_population_growth / 100 / 12 * months)
        target_employment = np.outer(pr_range, 1 - ur_range) * projected_population
        data = (target_employment - self.last_employment) / months * source_ratio

        return pd.DataFrame(
            data=data.round().astype(int),
            index=[f"{v:.1%}" for v in pr_range],
            columns=[f"{v:.1%}" for v in ur_range],
        ).rename_axis(index='participation_rate', columns='unemployment_rate')

    def extrapolate_unemployment_rate(
            self,
            months: int,
            source_ratio: float,
            net_migration_growth_estimate: float,
            trend_rate: float,
    ) -> float:
        annual_population_growth = net_migration_growth_estimate + self.birth_death_model
        projected_population = self.last_population * (1 + annual_population_growth / 100 / 12 * months)
        projected_labour_force = projected_population * self.last_participation
        projected_employment = self.last_employment + (trend_rate / source_ratio) * months
        return 1 - (projected_employment / projected_labour_force)


class UnemploymentSector:
    sectors = [
        'mining',
        'construction',
        'manufacturing',
        'trade & utilities',
        'information',
        'financial services',
        'professional services',
        'education & healthcare',
        'leisure & hospitality',
        'other services',
        'government',
    ]

    def __init__(self, df: pd.DataFrame):
        self.payrolls = df[[x + ' payrolls' for x in self.sectors]]
        self.openings = df[[x + ' openings' for x in self.sectors]]
        self.unemployed = seasonal_adjust(df[[x + ' unemployed' for x in self.sectors]], period=12)

    def calculate_sector_slack(self) -> pd.DataFrame:
        df = pd.DataFrame(
            self.openings.values / self.unemployed.values,
            index=self.openings.index,
            columns=self.sectors,
        )
        return df

    def calculate_payroll_growth(self) -> pd.DataFrame:
        pass


if __name__ == '__main__':
    unemployment_data = pd.read_csv(
        'assets/raw.csv',
        parse_dates=['dates'],
        dayfirst=True,
        index_col='dates',
        dtype=float,
    )

    sahm = calculate_sahm_rule(unemployment_data)
    sahm_trigger = extrapolate_sahm_trigger(sahm)



