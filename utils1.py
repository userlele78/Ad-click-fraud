import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker

class UnivariateAnalysis:
    def __init__(self, data): 
        '''
        Constructor method to initialize the UnivariateAnalysis object.

        Parameters:
        - data: DataFrame, the input data for analysis.
        '''
        self.data = data

    def visualize(self, x = str, width=20, height=8, rotate = None, create_other=False):
        '''
        Visualize the distribution of a categorical variable.

        Parameters:
        - x: str, the name of the categorical column to visualize.
        - width: int, width of the figure.
        - height: int, height of the figure.
        - rotate: int, rotation angle for x-axis labels.
        - create_other: bool, whether to combine small categories into an 'other' category.
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))
        self.cnt = self.data[x].value_counts()
        total_count = self.cnt.sum()

        # Create 'other' category if specified
        if create_other == True:
          threshold_percentage = 2
          mask = self.cnt / total_count * 100 < threshold_percentage
          other_count = self.cnt[mask].sum()
          self.cnt = self.cnt[~mask]
          self.cnt['other'] = other_count

        # Plot pie chart
        colors = sns.color_palette("Reds", len(self.cnt))
        ax1.pie(self.cnt, autopct='%.2f%%', labels=self.cnt.index, colors=colors)

        #Plot bar chart
        ax2.bar(self.cnt.index, self.cnt.values, color= colors)

        # Customize x-axis labels rotation
        if rotate != None:
            for tick in ax2.get_xticklabels():
                tick.set_rotation(rotate)

        # Adjust x-axis ticks for binary variables
        if self.data[x].nunique() == 2:
            if 0 in self.data[x].unique() and 1 in self.data[x].unique():
                ax2.set_xticks([0,1])
                ax2.set_xticklabels(self.cnt.index)
        plt.suptitle(f'Distribution of {x}')
        plt.show()

    def visualize_numeric(self, x = str, width=20, height=10, kde=False, common_bins=False):
        '''
        Visualize the distribution of a numeric variable.

        Parameters:
        - x: str, the name of the numeric column to visualize.
        - width: int, width of the figure.
        - height: int, height of the figure.
        - kde: bool, whether to plot kernel density estimation.
        - common_bins: bool, whether to use common bin edges in histograms.
        '''
        print(self.data[x].describe())
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height))
        sns.histplot(data=self.data, x=x, ax=ax1, kde=kde, common_bins=common_bins, color='#d62728')
        sns.boxplot(data=self.data, x=x, ax=ax2, color ='#d62728')
        fig.suptitle(f'Distribution of {x}')
        plt.show()


class StatisticAnalysis:
    def __init__(self, data ):
        '''
        Constructor method to initialize the StatisticAnalysis object.

        Parameters:
        - data: DataFrame, the input data for analysis.
        '''
        self.data = data

    def check_null(self, width=12, height=5):
        '''
        Check and visualize missing values in the dataset.

        Parameters:
        - width: int, width of the bar plot.
        - height: int, height of the bar plot.
        '''
        # Display the count of missing values for each column
        print(self.data.isnull().sum())

        # Calculate the percentage of missing values for each column
        null_df = self.data.isnull().sum() / self.data.shape[0] * 100
        null_df = null_df[null_df != 0].sort_values(ascending=False).reset_index()
        null_df.columns = ['feature', 'null_percentage']

        # Create a bar plot showing the percentage of missing values per column
        plt.figure(figsize=(width, height))
        custom_palette = sns.color_palette("Reds", len(null_df))
        sns.barplot(x=null_df['null_percentage'], y=null_df['feature'], palette=custom_palette, saturation=1,
                    order=null_df['feature'])
        plt.title('Null percentage per column', fontsize=20)
        plt.show()

    def correlation(self, width=12, height=10, drop_cols=None):
        '''
        Visualize the correlation matrix for numeric columns in the dataset.

        Parameters:
        - width: int, width of the heatmap.
        - height: int, height of the heatmap.
        - drop_cols: list, columns to be excluded from the correlation analysis.
        '''
        # Extract numeric columns from the dataset
        num_df = self.data._get_numeric_data()

        # Drop specified columns from the numeric dataframe
        if drop_cols is not None:
            num_df.drop(columns=drop_cols, inplace=True)

        # Calculate the correlation matrix
        corr_matrix = num_df.corr()

        # Create a heatmap of the correlation matrix
        plt.figure(figsize=(width, height))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=True,
                    fmt=".2f", linewidths=0.5,
                    mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
        plt.title('Correlation Heatmap')
        plt.show()


class BivariateAnalysis:
    def __init__(self):
        pass

    def scatter_plot(self, df1 = pd.DataFrame,
                     df2=pd.DataFrame, name_x= str, name_y=str,
                     width=12, height=6):
        '''
        Create a scatter plot for two dataframes based on specified columns.

        Parameters:
        - df1: DataFrame, data for non-defaulters (target = 0).
        - df2: DataFrame, data for defaulters (target = 1).
        - name_x: str, the column name for the x-axis.
        - name_y: str, the column name for the y-axis.
        - width: int, width of the figure.
        - height: int, height of the figure.
        '''
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(width, height))
        for ax in (ax1, ax2):
            ax.set_xlabel(name_x)
            ax.set_ylabel(name_y)

        ax1.scatter(df1[name_x], df1[name_y])
        ax1.set_title('Non-defaulter')
        ax1.set_xlim(list(ax1.get_xlim()))
        ax1.set_xticks([x+50000 for x in range(int(ax1.get_xlim()[0]), int(ax1.get_xlim()[1]), 50000)])
        ax1.set_xticklabels([str(x//1000) + 'k' for x in ax1.get_xticks()])

        ax2.scatter(df2[name_x], df2[name_y],color= 'red')
        ax2.set_title('Defaulter')
        ax2.set_xlim(list(ax2.get_xlim()))
        ax2.set_xticks([x+50000 for x in range(int(ax2.get_xlim()[0]), int(ax2.get_xlim()[1]), 50000)])
        ax2.set_xticklabels([str(x//1000) + 'k' for x in ax2.get_xticks()]);

        plt.show()

    def hist_plot(self, x=str, y = None, df1=pd.DataFrame, df2=pd.DataFrame, width=12, height=6, bins='auto', kde=False,
                  color='#eb0524'):
        '''
        Create histogram plots for non-defaulters and defaulters.

        Parameters:
        - x: str, the column name for the x-axis.
        - y: str, the column name for the y-axis.
        - df1: DataFrame, data for non-defaulters.
        - df2: DataFrame, data for defaulters.
        - width: int, width of the figure.
        - height: int, height of the figure.
        - bins: int or str, number of bins or binning strategy.
        - kde: bool, whether to plot kernel density estimation.
        - color: str, color of the histograms.
        '''
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(width, height))
        sns.histplot(df1, x=x, y=y, ax=ax1, bins=bins, kde=kde, color=color)
        ax1.set_title('Non-defaulter')
        sns.histplot(df2, x=x, y=y, ax=ax2, bins=bins, kde=kde, color=color)
        ax2.set_title('Defaulter')
        plt.show()

    def bar_plot(self, x=str, df1=pd.DataFrame, df2=pd.DataFrame, width=12, height=6, rotation=None):
        '''
        Create bar plots for non-defaulters and defaulters.

        Parameters:
        - x: str, the column name for the x-axis.
        - df1: DataFrame, data for non-defaulters.
        - df2: DataFrame, data for defaulters.
        - width: int, width of the figure.
        - height: int, height of the figure.
        - rotation: int, rotation angle for x-axis labels.
        '''
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(width, height))
        nd = df1[x].value_counts().sort_values()
        d = df2[x].value_counts().sort_values()

        ax1.bar(nd.index, nd.values, color = 'grey')
        ax1.set_title('Non-defaulter')

        ax2.bar(d.index, d.values, color='red')
        ax2.set_title('Defaulter')

        if rotation != None:
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=rotation)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=rotation)

        plt.show()

   
    def box_plot(self, x='is_attributed', y=str, data_to_plot=pd.DataFrame):
      '''
        Create a box plot for a specified column and target variable.

        Parameters:
        - x: str, the target variable column name.
        - y: str, the column name for the y-axis.
        - data_to_plot: DataFrame, data for plotting.
      '''
      sns.boxplot(x='is_attributed', y=y, data=data_to_plot, palette='Reds')
      plt.title("Box-Plot of {}".format(y))
      plt.show()

    def box_plot2(self, x=str, y=None, df1=pd.DataFrame,df2=pd.DataFrame,
                 width=12, height=6):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize= (width,height))
        sns.boxplot(data= df1, x=x, y=y, ax= ax1, color='red').set(title = "Non-defaulter")
        sns.boxplot(data= df2, x=x, y=y, ax= ax2, color='grey').set(title = "Defaulter")
        plt.show()

    def pie_plot(self, x=str, y=None, df1=pd.DataFrame,df2=pd.DataFrame,
                 width=12, height=6):
        '''
        Create pie charts for non-defaulters and defaulters.

        Parameters:
        - x: str, the column name for creating the pie chart.
        - y: None, not used in this method.
        - df1: DataFrame, data for non-defaulters.
        - df2: DataFrame, data for defaulters.
        - width: int, width of the figure.
        - height: int, height of the figure.
        '''

        fig, (ax1, ax2) = plt.subplots(1,2, figsize= (width,height))
        cnt1 = df1[x].value_counts()
        colors1 = sns.color_palette("Reds", len(cnt1))
        ax1.pie(cnt1, autopct='%.2f%%', labels=cnt1.index, colors=colors1)
        ax1.set_title('Non-defaulter')

        cnt2 = df2[x].value_counts()
        colors2 = sns.color_palette("Reds", len(cnt2))
        ax2.pie(cnt2, autopct='%.2f%%', labels=cnt2.index, colors=colors2)
        ax2.set_title('Defaulter')


    def percentage_of_defauter_per_cat(self, df=pd.DataFrame, col_name=str):
        '''
        Calculate and display the percentage of defaulters for each category.

        Parameters:
        - df: DataFrame, the input data.
        - col_name: str, the column for which to calculate the percentage.
        '''

        summary = []
        for cat in df[col_name].unique():
            default_count = df[(df[col_name] == cat) & (df.is_attributed == 1)].shape[0]
            total_count = df[df[col_name] == cat].shape[0]
            if total_count == 0:
                pass
            else:
                summary.append([cat ,default_count * 100 / total_count])

        report_df = pd.DataFrame(summary)
        report_df.columns = ["Categories", "Percentage_Of_Default"]
        report_df.sort_values(by='Percentage_Of_Default', ascending=False, inplace=True)

        sns.barplot(report_df, x='Percentage_Of_Default', y='Categories', palette='coolwarm')
        plt.title(col_name)
        plt.show()

def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    """Create a new column for each categorical value in categorical columns. """
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns

def do_sum(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    return df

def plot_categorical_variables_bar(data, column_name, figsize = (18,6), percentage_display = True, plot_defaulter = True, rotation = 0, 
                                   horizontal_adjust = 0, fontsize_percent = 'xx-small', color=list):
    '''
    Function to plot Categorical Variables Bar Plots

    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default = (18,6)
            Size of the figure to be plotted
        percentage_display: bool, default = True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default = 0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default = 'xx-small'
            Fontsize for percentage Display

    '''
    print(f"Total Number of unique categories of {column_name} = {len(data[column_name].unique())}")

    plt.figure(figsize = figsize, tight_layout = False)
    sns.set(style = 'whitegrid', font_scale = 1.2)

    #plotting overall distribution of category
    plt.subplot(1,2,1)
    data_to_plot = data[column_name].value_counts().sort_values(ascending = False)
    ax = sns.barplot(x = data_to_plot.index, y = data_to_plot, palette = color)

    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax.patches:
            ax.text(p.get_x() + horizontal_adjust, p.get_height() + 0.005 * total_datapoints, '{:1.02f}%'.format(p.get_height() * 100 / total_datapoints), fontsize = fontsize_percent)

    plt.xlabel(column_name, labelpad = 10)
    plt.title(f'Distribution of {column_name}', pad = 20)
    plt.xticks(rotation = rotation)
    plt.ylabel('Counts')

    #plotting distribution of category for Defaulters
    if plot_defaulter:
        percentage_defaulter_per_category = (data[column_name][data.is_attributed == 1].value_counts() * 100 / data[column_name].value_counts()).dropna().sort_values(ascending = False)

        plt.subplot(1,2,2)
        sns.barplot(x = percentage_defaulter_per_category.index, y = percentage_defaulter_per_category, palette = color)
        plt.ylabel('Percentage of Defaulter per category')
        plt.xlabel(column_name, labelpad = 10)
        plt.xticks(rotation = rotation)
        plt.title(f'Percentage of Defaulters for each category of {column_name}', pad = 20)
    plt.show()

def missing_values_table(df):

    """Function to calculate missing values by column."""

    # Total missing values
    miss_val = df.isnull().sum()

    # Percentage of missing values
    miss_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    miss_val_table = pd.concat([miss_val, miss_val_percent], axis=1)

    # Rename the columns
    miss_val_table_ren_columns = miss_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    miss_val_table_ren_columns = miss_val_table_ren_columns[
        miss_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
           "There are " + str(miss_val_table_ren_columns.shape[0]) + " columns that have missing values.")

    # Return the dataframe with missing information
    return miss_val_table_ren_columns

def print_unique_categories(data, column_name, show_counts = False):

    '''
    Hiển thị thông tin cơ bản về biến phân loại (categorical variable) trong DataFrame, bao gồm danh sách các giá trị duy nhất và (tùy chọn) số lượng của từng giá trị.

    Parameters:
        data: DataFrame
            DataFrame chứa dữ liệu.
        column_name: str
            Tên của cột (biến) mà bạn muốn hiển thị thống kê.
        show_counts: bool, mặc định là False.
            Nếu True, hàm sẽ hiển thị cả số lượng (counts) của mỗi giá trị.
    '''

    print('-'*100)
    print(f"The unique categories of '{column_name}' are: \n{data[column_name].unique()}")
    print('-'*100)

    if show_counts:
        print(f"Counts of each category are: \n{data[column_name].value_counts()}")
        print('-'*100)

def calculate_perc_categories(df, groupby_col, target_col):

    """
    Tính % của các giá trị trong cột 'target_column' theo các nhóm trong 'groupby_columns'.

    Parameters:
        df: DataFrame
            DataFrame chứa dữ liệu.
        groupby_col: (list)
            Danh sách các cột dùng để nhóm.
        target_col (str): Tên cột chứa giá trị 'TARGET'.

    Returns:
    - pd.DataFrame: DataFrame chứa tỷ lệ phần trăm của 'target_col' theo 'groupby_col'.
    """

    # Nhóm dữ liệu và tính tỷ lệ phần trăm
    grouped_data = df.groupby(groupby_col + [target_col]).size().unstack(fill_value=0)
    percentage_data = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100

    return percentage_data

def format_thousands(value, _):
    """
    Định dạng số hàng nghìn với dấu phẩy.
    """
    return "{:,.0f}".format(value)

def plot_categorical_bar_horizontal(data, column_name, figsize=(18, 8), percentage_display=True, 
                                    plot_defaulter=True, rotation=0, 
                                    horizontal_adjust=0.25, fontsize_percent='xx-small'):

    '''
    Function to plot Categorical Variables Bar Plots

    Parameters:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default=(18,6)
            Size of the figure to be plotted
        percentage_display: bool, default=True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default=0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default=0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default='xx-small'
            Fontsize for percentage Display
    '''

    print(f"Total Number of unique categories of {column_name} = {len(data[column_name].unique())}")

    plt.figure(figsize=figsize, tight_layout=False)
    sns.set(style='whitegrid', font_scale=1.2)

    custom_palette = ['#eb0524', '#d33944', '#de6b73', '#e4848a', '	#e99ca1', '#efb5b9', '#f4ced0', '#fae6e8']

    # Plotting overall distribution of category
    plt.subplot(1, 2, 1)
    data_to_plot = data[column_name].value_counts().sort_values(ascending=False)
    ax1 = sns.barplot(x=data_to_plot, y=data_to_plot.index.tolist(), palette=sns.color_palette('Paired'), orient='h')

    # ax1 = sns.barplot(x=data_to_plot, y=data_to_plot.index, palette=custom_palette, orient='h')

    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax1.patches:
            ax1.text(p.get_width() + horizontal_adjust + 1000, p.get_y() + p.get_height() / 2, '{:1.02f}%'.format(p.get_width() * 100 / total_datapoints), fontsize=fontsize_percent, va='center')

    # remove spines
    ax1.spines[['right', 'left', 'top']].set_visible(False)  #xoá spine
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=rotation)
    ax1.xaxis.set_major_formatter(FuncFormatter(format_thousands))

    # plt.ylabel(column_name, labelpad=10)
    plt.title(f'Distribution of {column_name}', pad=20)
    plt.xlabel('')

    # Plotting distribution of category for Defaulters
    if plot_defaulter:
        percentage_defaulter_per_category = (data[column_name][data.is_attributed == 1].value_counts() * 100 / data[column_name].value_counts()).dropna().sort_values(ascending=False)

        plt.subplot(1, 2, 2)
        ax2 = sns.barplot(x=percentage_defaulter_per_category, y=percentage_defaulter_per_category.index, 
                          palette=sns.color_palette('Paired'), orient='h')
        plt.xlabel('Percentage of Defaulter per category')
        plt.ylabel(column_name, labelpad=10)
        plt.title(f'Percentage of Defaulters for each category of {column_name}', pad=20)

        # Annotate for ax2
        for p in ax2.patches:
            ax2.text(p.get_width() + horizontal_adjust, p.get_y() + p.get_height() / 2, '{:.2f}%'.format(p.get_width()), fontsize=fontsize_percent, va='center')

        # remove spines
        ax2.spines[['right', 'left', 'top']].set_visible(False)  # Xoá spines
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=rotation)

    plt.show()

def plot_categorical_bar_vertical(data, column_name, figsize=(18, 8), 
                                  percentage_display=True, plot_defaulter=True, rotation=0, 
                                  horizontal_adjust=0.25, fontsize_percent='xx-small'):
    '''
    Function to plot Categorical Variables Bar Plots

    Parameters:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default=(18,6)
            Size of the figure to be plotted
        percentage_display: bool, default=True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default=0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default=0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default='xx-small'
            Fontsize for percentage Display
    '''

    print(f"Total Number of unique categories of {column_name} = {len(data[column_name].unique())}")

    plt.figure(figsize=figsize, tight_layout=False)
    sns.set(style='whitegrid', font_scale=1.2)
    custom_palette = ['#eb0524', '#B4B4B3']

    # Plotting overall distribution of category
    plt.subplot(1, 2, 1)
    data_to_plot = data[column_name].value_counts().sort_values(ascending=False)
    ax1 = sns.barplot(x=data_to_plot.index, y=data_to_plot, palette=sns.color_palette('Paired'), orient='v')

    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax1.patches:
            ax1.text(p.get_x() + p.get_width() / 2,
                     p.get_height() + horizontal_adjust + 4000,
                     '{:1.02f}%'.format(p.get_height() * 100 / total_datapoints),
                     fontsize=fontsize_percent,
                     ha='center')

    # remove spines
    ax1.spines[['right', 'left', 'top']].set_visible(False)  # Xoá spines
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=rotation)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_thousands))

    plt.title(f'Distribution of {column_name}', pad=20)
    plt.ylabel('')
    plt.xlabel(column_name)

    # Plotting distribution of category for Defaulters
    if plot_defaulter:
        percentage_defaulter_per_category = (data[column_name][data.is_attributed == 1].value_counts() * 100 / data[column_name].value_counts()).dropna().sort_values(ascending=False)

        plt.subplot(1, 2, 2)
        ax2 = sns.barplot(x=percentage_defaulter_per_category.index, y=percentage_defaulter_per_category, palette=sns.color_palette('Paired'), orient='v')  # Thay đổi orient thành 'v'
        plt.ylabel('Percentage of Defaulter per category')
        plt.xlabel(column_name)
        plt.title(f'Percentage of Defaulters for each category of {column_name}', pad=20)

        # Annotate for ax2
        for p in ax2.patches:
            ax2.text(p.get_x() + p.get_width() / 2, p.get_height() + horizontal_adjust, '{:.2f}%'.format(p.get_height()), fontsize=fontsize_percent, ha='center')

        # remove spines
        ax2.spines[['right', 'left', 'top']].set_visible(False)  # Xoá spines
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=rotation)

    plt.show()

def plot_stats(data, feature, label_rotation=False, horizontal_layout=True):
    # Set the matplotlib color cycle to a custom list of colors
    colors = ['#eb0524', '#c4c4c4']

    # Count the occurrences of each unique value in the specified feature
    temp = data[feature].value_counts()

    # Create a DataFrame from the counts
    df1 = pd.DataFrame({feature: temp.index, 'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = data[[feature, 'is_attributed']].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by='is_attributed', ascending=False, inplace=True)

    # Create subplots based on the specified layout   (18, 8)
    if horizontal_layout:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(18, 10))

    # Plot a bar chart for the number of contracts in the first subplot
    ax1.bar(df1[feature], df1['Number of contracts'], color=colors, edgecolor='black')

    # Optionally rotate the x-axis labels based on the provided argument
    if label_rotation:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    # Plot a bar chart for the percentage of target with value 1 in the second subplot
    ax2.bar(cat_perc[feature], cat_perc['is_attributed'], color=colors, edgecolor='black')

    # Optionally rotate the x-axis labels based on the provided argument
    if label_rotation:
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

    # Set labels and tick parameters for both subplots
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    ax1.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)

    plt.subplots_adjust(wspace=2)

    # Display the plots
    plt.suptitle(f'{feature}', fontsize=25, fontweight='bold', color='black')
    plt.show();

def plot_distribution(data, column):

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    colors = ['#dc143c', '#c4c4c4']

    # Pie chart
    labels = data[column].value_counts().index
    vals = data[column].value_counts().values

    ax[0].pie(vals,
              explode=[0, 0.2],
              labels=labels,
              colors=colors,
              autopct='%.2f%%',
              shadow=False,
              wedgeprops=dict(edgecolor='black'))

    # Chỉnh lại size của annotate
    for text in ax[0].texts:
        text.set_fontsize(12)

    ax[0].set_ylabel('')

    # Countplot
    bars = ax[1].bar(labels,
                    vals,
                    color=colors,
                    edgecolor='black')

    ax[1].set_xticks([0.00, 1.00])
    ax[1].set_yticks([])  # Xoá ytick
    ax[1].spines[['right', 'left', 'top']].set_visible(False)  # Xoá spine bên phải

    for bar in bars:
        count = bar.get_height()
        formatted_count = '{:,.0f}'.format(count)
        ax[1].annotate(f'{formatted_count}',
                       xy=(bar.get_x() + bar.get_width() / 2, count),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, color='black')

    # Chỉnh size của xtick
    ax[0].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='x', labelsize=16)

    # Chỉnh khoảng cách giữa 2 ax
    plt.subplots_adjust(wspace=0.8)

    # Show the plot
    plt.suptitle(f'The Distribution of {column} value', fontsize=25, fontweight='bold', color='black')

def plot_histogram(data, column, title, xlabel):
    """
    Vẽ biểu đồ histogram cho 1 cột dữ liệu cụ thể trong DataFrame.

    Parameters:
        data: DataFrame
            DataFrame chứa dữ liệu.
        column: str
            Tên của cột dữ liệu mà bạn muốn vẽ histogram
        title: str
            Tên của biểu đồ Histogram.
        xlabel: str
            Label của trục x.
    """

    plt.figure(figsize=(10, 6))
    # plt.style.use('default')
    ax = data[column].plot.hist(title=title,
                                color='#dc143c',
                                edgecolor='white')

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))  #định dạng dấu phẩy cho trục
    ax.yaxis.grid(True)  #hiển thị grid ngang
    ax.set_ylabel('')  # Xoá nhãn trục y
    ax.spines[['right', 'left', 'top']].set_visible(False)  #xoá spine
    ax.set_title(title, fontsize=20, fontweight='bold') #chỉnh title

    plt.xlabel(xlabel)
    plt.show()

def plot_continuous_variables(data, column_name, plots = ['distplot', 'CDF', 'box', 'violin'], scale_limits = None, figsize = (20,8), histogram = True, log_scale = False):

    '''
    Function to plot continuous variables distribution

    Inputs:
        data: DataFrame
            The DataFrame from which to plot.
        column_name: str
            Column's name whose distribution is to be plotted.
        plots: list, default = ['distplot', 'CDF', box', 'violin']
            List of plots to plot for Continuous Variable.
        scale_limits: tuple (left, right), default = None
            To control the limits of values to be plotted in case of outliers.
        figsize: tuple, default = (20,8)
            Size of the figure to be plotted.
        histogram: bool, default = True
            Whether to plot histogram along with distplot or not.
        log_scale: bool, default = False
            Whether to use log-scale for variables with outlying points.
    '''

    data_to_plot = data.copy()
    if scale_limits:
        #taking only the data within the specified limits
        data_to_plot[column_name] = data[column_name][(data[column_name] > scale_limits[0]) & (data[column_name] < scale_limits[1])]

    number_of_subplots = len(plots)
    plt.figure(figsize = figsize)
    sns.set_style('whitegrid')

    for i, ele in enumerate(plots):
        plt.subplot(1, number_of_subplots, i + 1)
        plt.subplots_adjust(wspace=0.25)

        if ele == 'CDF':
            #making the percentile DataFrame for both positive and negative Class Labels
            percentile_values_0 = data_to_plot[data_to_plot.is_attributed == 0][[column_name]].dropna().sort_values(by = column_name)
            percentile_values_0['Percentile'] = [ele / (len(percentile_values_0)-1) for ele in range(len(percentile_values_0))]

            percentile_values_1 = data_to_plot[data_to_plot.is_attributed == 1][[column_name]].dropna().sort_values(by = column_name)
            percentile_values_1['Percentile'] = [ele / (len(percentile_values_1)-1) for ele in range(len(percentile_values_1))]

            plt.plot(percentile_values_0[column_name], percentile_values_0['Percentile'], color = '#eb0524', label = 'Non-Defaulters')
            plt.plot(percentile_values_1[column_name], percentile_values_1['Percentile'], color = 'black', label = 'Defaulters')
            plt.xlabel(column_name)
            plt.ylabel('Probability')
            plt.title('CDF of {}'.format(column_name))
            plt.legend(fontsize = 'medium')
            if log_scale:
                plt.xscale('log')
                plt.xlabel(column_name + ' - (log-scale)')

        if ele == 'distplot':
            sns.distplot(data_to_plot[column_name][data['is_attributed'] == 0].dropna(),
                         label='Non-Defaulters', hist = False, color='#eb0524')
            sns.distplot(data_to_plot[column_name][data['is_attributed'] == 1].dropna(),
                         label='Defaulters', hist = False, color='black')
            plt.xlabel(column_name)
            plt.ylabel('Probability Density')
            plt.legend(fontsize='medium')
            plt.title("Dist-Plot of {}".format(column_name))
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{column_name} (log scale)')

        if ele == 'violin':
            sns.violinplot(x='is_attributed', y=column_name, data=data_to_plot)
            plt.title("Violin-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

        if ele == 'box':
            sns.boxplot(x='is_attributed', y=column_name, data=data_to_plot, palette=['#eb0524', 'black'])
            plt.title("Box-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

    plt.show()