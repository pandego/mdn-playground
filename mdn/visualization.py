import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly import offline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def save_plot_as_png_and_html(fig, filename, width=1200, height=800, scale=2):
    pio.write_image(fig, filename + ".png", width=width, height=height, scale=scale)
    offline.plot(fig, filename=filename + ".html", auto_open=False)


########################################
### --- Simple Linear Regression --- ###
########################################
def plot_train_val_data(x_train, y_train, x_val, y_val, run_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train.flatten(), y=y_train.flatten(), mode='markers',
                             name='Train Data', marker=dict(color='blue', opacity=0.5)))
    fig.add_trace(go.Scatter(x=x_val.flatten(), y=y_val.flatten(), mode='markers',
                             name='Validation Data',
                             marker=dict(color='orange', opacity=0.5)))
    fig.update_layout(title='Train and Validation Data', xaxis_title='x',
                      yaxis_title='y')
    save_plot_as_png_and_html(fig, f"{run_path}/train_val_data")


def plot_conditional_mode(x_train, y_train, x_test, cond_mode, run_path):
    title = "Conditional Mode"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train.flatten(), y=y_train.flatten(), mode='markers',
                             name='Training Data',
                             marker=dict(color='blue', opacity=0.5)))
    fig.add_trace(go.Scatter(x=x_test.flatten(), y=cond_mode.detach().numpy().flatten(),
                             mode='markers', name='Predictions',
                             marker=dict(color='red')))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    save_plot_as_png_and_html(fig, f"{run_path}/{title.replace(' ', '_')}")


def plot_means(x_train, y_train, x_test, mu, num_mixtures, run_path):
    title = "Means"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train.flatten(), y=y_train.flatten(), mode='markers',
                             name='Training Data',
                             marker=dict(color='blue', opacity=0.5)))
    for i in range(num_mixtures):
        fig.add_trace(
            go.Scatter(x=x_test.flatten(), y=mu.detach().numpy()[:, i].flatten(),
                       mode='markers', name=f'Mean {i + 1}',
                       marker=dict(color='red', opacity=0.3)))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    save_plot_as_png_and_html(fig, f"{run_path}/{title.replace(' ', '_')}")


def plot_sampled_predictions(x_train, y_train, x_test, preds, run_path):
    title = "Sampled Predictions"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train.flatten(), y=y_train.flatten(), mode='markers',
                             name='Training Data',
                             marker=dict(color='blue', opacity=0.5)))
    for i in range(preds.shape[1]):
        fig.add_trace(
            go.Scatter(x=x_test.flatten(), y=preds[:, i, :].detach().numpy().flatten(),
                       mode='markers', name=f'Prediction {i + 1}',
                       marker=dict(color='red', opacity=0.3)))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    save_plot_as_png_and_html(fig, f"{run_path}/{title.replace(' ', '_')}")


##########################################
### --- Multiple Linear Regression --- ###
##########################################
def plot_histogram(y_pred, y_test, run_path, target_column):
    y_pred_series = pd.Series(y_pred.flatten())
    y_test_series = pd.Series(y_test.flatten())
    nbinsx = 100

    hist_pred = go.Histogram(
        x=y_pred_series,
        nbinsx=nbinsx,
        name='Predictions',
        opacity=1
    )
    hist_test = go.Histogram(
        x=y_test_series,
        nbinsx=nbinsx,
        name=f'{target_column} Measurements',
        opacity=1
    )

    layout = go.Layout(
        title=f'Histogram of {target_column} Measured VS Model Prediction',
        xaxis=dict(title='Values'),
        yaxis=dict(title='Frequency'),
        # barmode='overlay',    # uncomment to overlay histograms
        # barmode='group',      # uncomment to display histograms side by side
    )

    fig = go.Figure(data=[hist_pred, hist_test], layout=layout)
    save_plot_as_png_and_html(fig, f"{run_path}/histo_y_true_vs_y_pred")


def plot_scatter(y_pred, y_test, run_path, target_column):
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    fig = px.scatter(
        x=y_test.flatten(),
        y=y_pred.flatten(),
        trendline="ols",
        labels={"x": "Measured", "y": "Predicted"},
        title=f'Scatter Plot of {target_column} Measured VS Model Prediction '
              f'(R²={r_squared:.2f}, MAE={mae:.2f}, MSE={mse:.2f})'
    )
    fig.data[1].line.color = 'red'

    save_plot_as_png_and_html(fig, f"{run_path}/scatter_y_true_vs_y_pred")


###################################
# --- Multivariate Regression --- #
###################################
def plot_multivariate_regression(x_train, y_train, x_test, y_test, y_pred, run_path):
    NotImplementedError("Multivariate regression plot not implemented yet.")
    # TODO: Add plot for multivariate regression
