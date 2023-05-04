import numpy as np
import math as math
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly

# Random seed to ensure random results are repeated
"""np.random.seed(0)"""

# Portfolio Parameters input file path and result output files
parameters_file = './portfolio_parameters.xlsx'
csv_output_file = './sim_results_quantiles.csv'
sim_output_file = './sim_results_details.csv'
plot_pdf_file = './sim_results_plot.pdf'

# Assign portfolio parameters to a dataframe and then to data types needed for the simulation
parameters = pd.read_excel(parameters_file)

# Month count, risk-neutral/no-arbitrage mean prices and daily sigmas
delivery_months = parameters.Month.count()
delivery_month_tags = parameters.Month.to_list()
mean_px = parameters.FWD_Mark.to_list()
daily_sigma = parameters.Daily_STDEV.to_list()

# Jump diffusion probability and jump sizes
jump_probability = parameters.Jump_Prob.to_list()
jump_size = parameters.Jump_Size.to_list()
correlation_matrix = parameters.iloc[:, 9:33].values

# Fundamental floor and ceiling prices
min_px = dict(parameters.loc[:, ['Month', 'Min_Px']].values)
max_px = dict(parameters.loc[:, ['Month', 'Max_Px']].values)

# Holding period in # of trading days
hold = 22

# Number of simulation runs
num_sims = 1000

# Define function to convert daily volatility to a longer time horizon:
def sigma_convert(daily_sigma, days_hold=22):
    return math.sqrt(days_hold) * daily_sigma

# Convert daily volatility from daily to hold-period provided
sigma = [sigma_convert(sigma, hold) for sigma in daily_sigma]

# Calculate covariance matrix. Not used anywhere but can be used in future versions of the code
covariance_matrix = np.outer(sigma, sigma) \
                    * correlation_matrix \
                    * np.outer(mean_px, mean_px)

forward_curve = np.zeros((num_sims, delivery_months))
for sim in range(num_sims):
    random_values = np.random.multivariate_normal(
        np.zeros(delivery_months),
        correlation_matrix)

    for month in range(delivery_months):
        price_change = np.exp(random_values[month] * sigma[month])
        price_jump = (
            1 + jump_size[month] if np.random.uniform(0, 1) < jump_probability[month] else 1
        )
        forward_curve[sim, month] = \
            max(min_px[delivery_month_tags[month]],
                min(mean_px[month] * price_change * price_jump,
                    max_px[delivery_month_tags[month]])
                )

# Convert numpy.ndarray to DataFrame for transformation and visualization
forward_curve = pd.DataFrame(forward_curve)
forward_curve["Simulation"] = range(num_sims)
forward_curve.columns = delivery_month_tags + ['Simulation']

forward_curve = forward_curve\
    .melt(id_vars="Simulation",
          var_name="Delivery_Month",
          value_name="Price")

# Calculate mean and p-levels and assign to dataframe for csv output
mean_values = forward_curve.groupby("Delivery_Month")['Price']\
    .mean()\
    .reset_index()\
    .rename({'Price': 'Mean Px'})
median_values = forward_curve.groupby("Delivery_Month")['Price']\
    .median()\
    .reset_index()\
    .rename({'Price': 'Median Px'})
p5_values = forward_curve.groupby("Delivery_Month")['Price']\
    .quantile(0.05)\
    .reset_index()
p95_values = forward_curve.groupby("Delivery_Month")['Price']\
    .quantile(0.95)\
    .reset_index()
p1_values = forward_curve.groupby("Delivery_Month")['Price']\
    .quantile(0.01)\
    .reset_index()
p99_values = forward_curve.groupby("Delivery_Month")['Price']\
    .quantile(0.99)\
    .reset_index()

df_list = [median_values, p95_values, p5_values, p99_values, p1_values]
dfs = mean_values.copy()
for df in df_list:
    dfs = pd.merge(dfs, df, on=['Delivery_Month'])
dfs.columns = ['Delivery_Month', 'Mean', 'Median', 'P95', 'P5', 'P99', 'P1']

################################## Plotly plots below ##########################################
# Create line graph of simulated forward curves using Plotly
fig = px\
    .line(forward_curve,
          x="Delivery_Month",
          y="Price",
          color="Simulation",
          line_group="Simulation",
          title="Forward Curve Simulation Results"
          )
fig.update_traces(
    mode="lines",
    line=dict(width=0.35, color="gray"), opacity=0.2,
    showlegend=False
)

# Calculate and plot the resulting mean of simulated curves
fig.add_trace(
    go.Scatter(x=mean_values["Delivery_Month"],
               y=mean_values["Price"],
               mode="lines",
               line=dict(color="black", width=5),
               opacity=0.8,
               name="mean")
)

# Calculate and plot the resulting P5 outcome
fig.add_trace(
    go.Scatter(x=p5_values["Delivery_Month"],
               y=p5_values["Price"],
               mode="lines",
               line=dict(color="green", width=3),
               opacity=0.8,
               name="P5")
)

# Calculate and plot the resulting P95 outcome
fig.add_trace(
    go.Scatter(x=p95_values["Delivery_Month"],
               y=p95_values["Price"],
               mode="lines",
               line=dict(color="red", width=3),
               opacity=0.8,
               name="P95")
)

# Prepare the plot for printing
fig.update_xaxes(nticks=24)
fig.update_xaxes(tickangle= -90)

# Generate a pdf image of the plot as well as csv output of the sim result mean & quantiles
plotly.io.write_image(fig, plot_pdf_file, format='pdf')
dfs.to_csv(csv_output_file, index=False)
forward_curve.to_csv(sim_output_file, index=False)

# Show the plot
fig.show()

