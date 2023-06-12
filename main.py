import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
from wordcloud import WordCloud
from sklearn.preprocessing import MultiLabelBinarizer 
import matplotlib
import seaborn as sns

hex_values = ["#009ca5", "#437376", "#4b4b4b"]

def get_dataframe(path: str) -> DataFrame:
    df = pd.read_csv(path)
    return df


def get_movie_tv_show_count_by_year(df: DataFrame):
    '''
    Computes an image with the distribution of movies/tv shows per year
    '''
    # Keep only release_year and type for the columns
    x = df[['release_year', 'type']]

    # Count the number of movies/tv shows per year
    movies = x[x.type=='Movie'].groupby(['release_year']).count()
    movies = movies.rename(columns={'type': 'movie'})
    tv_shows = x[x.type=='TV Show'].groupby(['release_year']).count()
    tv_shows = tv_shows.rename(columns={'type': 'tv_show'})

    # Concatenate the movies and tv_shows dataframes
    res = pd.concat([movies, tv_shows], axis=1)
    res = res.fillna(0)
    res = res.sort_index()

    # Create a new column for the years and reset the index
    res['year'] = res.index
    res = res.reset_index()

    # Calculate percentage of movies / tv_shows per year
    res['movie'] = res['movie'] / (res['movie'] + res['tv_show'])
    res['movie'] = round(res['movie'], 2)
    res['tv_show'] = 1.0 - res['movie']

    # Get only the last 15 years
    res = res.tail(15)

    # Plot the results
    plot_movie_tv_show_count_by_year(res)


def plot_movie_tv_show_count_by_year(res: DataFrame):
    fig, ax = plt.subplots(1,1,figsize=(6.5, 15.5))

    ax.barh(res.index, res['movie'], 
        color=hex_values[0], alpha=0.9, label='Movies')
    ax.barh(res.index, res['tv_show'], left=res['movie'], 
        color=hex_values[-1], alpha=0.9, label='TV Shows')
    
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # display year
    for i in res.index:
        ax.annotate(f"{res['year'][i]}", 
                    xy=(1/2, i),
                    va = 'center', ha='center', fontsize=16, fontweight='light', fontfamily='serif',
                    color='white')

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)

    ax.legend()

    path = 'images/movie_tv_show_count_by_year.png'
    plt.savefig(path)
    print(f"Image saved here: {path}")


def get_movie_tv_show_percentage(df: DataFrame):
    x = df.groupby(['type'])['type'].count()
    y = len(df)
    r = ((x/y)).round(2)

    mf_ratio = pd.DataFrame(r).T

    fig, ax = plt.subplots(1,1,figsize=(6.5, 2.5))

    ax.barh(mf_ratio.index, mf_ratio['Movie'], 
            color=hex_values[0], alpha=0.9, label='Male')
    ax.barh(mf_ratio.index, mf_ratio['TV Show'], left=mf_ratio['Movie'], 
            color=hex_values[-1], alpha=0.9, label='Female')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in mf_ratio.index:
        ax.annotate(f"{int(mf_ratio['Movie'][i]*100)}%", 
                    xy=(mf_ratio['Movie'][i]/2, i),
                    va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
                    color='white')
        ax.annotate("Movie", 
                    xy=(mf_ratio['Movie'][i]/2, -0.25),
                    va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
                    color='white')
                
    for i in mf_ratio.index:
        ax.annotate(f"{int(mf_ratio['TV Show'][i]*100)}%", 
                    xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, i),
                    va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
                    color='white')
        ax.annotate("TV Show", 
                    xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, -0.25),
                    va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
                    color='white')
        
        # Title & Subtitle
    fig.text(0.125,1.03,'Movie & TV Show distribution', fontfamily='serif',fontsize=15, fontweight='bold')

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)
        
    path = 'images/movie_tv_show_percentage.png'
    plt.savefig(path)
    print(f"Image saved here: {path}")


def get_top_countries(df: DataFrame):
    x = df
    x['count'] = 1
    
    x['country'] = x['country'].fillna(x['country'].mode()[0])
    # Many productions have several countries listed - this will skew our results , we'll grab the first one mentioned

    # Lets retrieve just the first country
    x['first_country'] = x['country'].apply(lambda x: x.split(",")[0])

    x['first_country'].replace('United States', 'USA', inplace=True)
    x['first_country'].replace('United Kingdom', 'UK',inplace=True)
    x['first_country'].replace('South Korea', 'S. Korea',inplace=True)

    data = x.groupby('first_country')['count'].sum().sort_values(ascending=False)[:10]

    color_map = [hex_values[-1] for _ in range(10)]
    color_map[0] = color_map[1] = color_map[2] =  hex_values[0] # color highlight

    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(data.index, data, width=0.5, 
        edgecolor='darkgray',
        linewidth=0.6,color=color_map)

    #annotations
    for i in data.index:
        ax.annotate(f"{data[i]}", 
                    xy=(i, data[i] + 150), #i like to change this to roughly 5% of the highest cat
                    va = 'center', ha='center',fontweight='light', fontfamily='serif')
        
    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    # Tick labels

    ax.set_xticklabels(data.index, fontfamily='serif', rotation=0)

    ax.grid(axis='y', linestyle='-', alpha=0.4)   

    grid_y_ticks = np.arange(0, 4000, 500) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    ax.set_axisbelow(True)
        
    # thicken the bottom line if you want to
    plt.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

    ax.tick_params(axis='both', which='major', labelsize=12)


    import matplotlib.lines as lines
    l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig,color='black',lw=0.2)
    fig.lines.extend([l1])

    ax.tick_params(axis=u'both', which=u'both',length=0)

    path = 'images/top_countries.png'
    plt.savefig(path)
    print(f"Image saved here: {path}")


def get_content_by_country(df: DataFrame):
    country_order = df['first_country'].value_counts()[:11].index
    data_q2q3 = df[['type', 'first_country']].groupby('first_country')['type'].value_counts().unstack().loc[country_order]
    data_q2q3['sum'] = data_q2q3.sum(axis=1)
    data_q2q3_ratio = (data_q2q3.T / data_q2q3['sum']).T[['Movie', 'TV Show']].sort_values(by='Movie',ascending=False)[::-1]

    fig, ax = plt.subplots(1,1,figsize=(15, 8),)

    ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['Movie'], 
            color=hex_values[0], alpha=0.8, label='Movie')
    ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['TV Show'], left=data_q2q3_ratio['Movie'], 
            color=hex_values[-1], alpha=0.8, label='TV Show')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticklabels(data_q2q3_ratio.index, fontfamily='serif', fontsize=11)

    # male percentage
    for i in data_q2q3_ratio.index:
        ax.annotate(f"{data_q2q3_ratio['Movie'][i]*100:.3}%", 
                    xy=(data_q2q3_ratio['Movie'][i]/2, i),
                    va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
                    color='white')

    for i in data_q2q3_ratio.index:
        ax.annotate(f"{data_q2q3_ratio['TV Show'][i]*100:.3}%", 
                    xy=(data_q2q3_ratio['Movie'][i]+data_q2q3_ratio['TV Show'][i]/2, i),
                    va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
                    color='white')
        
    # fig.text(0.13, 0.93, 'Top 10 countries Movie & TV Show split', fontsize=15, fontweight='bold', fontfamily='serif')   
    # fig.text(0.131, 0.89, 'Percent Stacked Bar Chart', fontsize=12,fontfamily='serif')   

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)
        
    #ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))

    fig.text(0.75,0.9,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color=hex_values[0])
    fig.text(0.81,0.9,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.82,0.9,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color=hex_values[1])

    import matplotlib.lines as lines
    l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig,color='black',lw=0.2)
    fig.lines.extend([l1])

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis=u'both', which=u'both',length=0)

    path = 'images/content_by_country.png'
    plt.savefig(path)
    print(f"Image saved here: {path}")


def get_content_by_month(df: DataFrame):
    df['date_added'] = df['date_added'].str.strip()
    df["date_added"] = pd.to_datetime(df['date_added'])

    df['month_added']=df['date_added'].dt.month
    df['month_name_added']=df['date_added'].dt.month_name()
    df['year_added'] = df['date_added'].dt.year
    month_order = ['January',
                    'February',
                    'March',
                    'April',
                    'May',
                    'June',
                    'July',
                    'August',
                    'September',
                    'October',
                    'November',
                    'December']

    df['month_name_added'] = pd.Categorical(df['month_name_added'], categories=month_order, ordered=True)

    data_sub = df.groupby('type')['month_name_added'].value_counts().unstack().fillna(0).loc[['TV Show','Movie']].cumsum(axis=0).T

    data_sub2 = data_sub

    data_sub2['Value'] = data_sub2['Movie'] + data_sub2['TV Show']
    data_sub2 = data_sub2.reset_index()

    df_polar = data_sub2.sort_values(by='month_name_added', ascending=False)
    df_polar = data_sub2.sort_values(by='Value', ascending=False)

    color_map = [hex_values[-1] for _ in range(12)]
    color_map[10] = color_map[11] = hex_values[0] # color highlight


    # initialize the figure
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')

    # Constants = parameters controling the plot layout:
    upperLimit = 30
    lowerLimit = 1
    labelPadding = 30

    # Compute max and min in the dataset
    max = df_polar['Value'].max()

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * df_polar.Value + lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(df_polar.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df_polar.index)+1))
    angles = [element * width for element in indexes]
    angles

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=2, 
        edgecolor="white",
        color=color_map,alpha=0.8
    )

    # Add labels
    for bar, angle, height, label in zip(bars, angles, heights, df_polar["month_name_added"]):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle, 
            y=lowerLimit + bar.get_height() + labelPadding, 
            s=label, 
            ha=alignment, fontsize=10,fontfamily='serif',
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor") 
        
    path = 'images/content_release_month.png'
    plt.savefig(path)
    print(f"Image saved here: {path}")


def get_wordcloud_from_titles(df: DataFrame):
    # Custom colour map based on Netflix palette
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [hex_values[0], hex_values[-1]])

    text = str(list(df['title'])).replace(',', '').replace('[', '').replace("'", '').replace(']', '').replace('.', '')

    wordcloud = WordCloud(background_color = 'white', width = 500,  height = 500,colormap=cmap, max_words = 150).generate(text)

    plt.figure( figsize=(5,5))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    path = 'images/word_cloud_titles.png'
    plt.savefig(path)
    print(f"Image saved here: {path}")


def get_genre_analysis_movie(df: DataFrame):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#00f1ff', '#151515'])
    df = df[df["type"] == "Movie"]
    df['genre'] = df['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 
    Types = []
    for i in df['genre']: 
        Types += i
    Types = set(Types)
    test = df['genre']
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)
    corr = res.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(8, 7))
    pl = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, vmin=-.3, center=0, square=True, linewidths=1.5)

    path = 'images/genre_corelation.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Image saved here: {path}")


def get_genre_analysis_tv_shows(df: DataFrame):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#00f1ff', '#151515'])
    df = df[df["type"] == "TV Show"]
    df['genre'] = df['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 
    Types = []
    for i in df['genre']: 
        Types += i
    Types = set(Types)
    test = df['genre']
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)
    corr = res.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(8, 7))
    pl = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, vmin=-.3, center=0, square=True, linewidths=1.5)

    path = 'images/genre_corelation_tv_show.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Image saved here: {path}")



def main():
    path = "dataset/netflix_titles.csv"
    df = get_dataframe(path)
    
    get_movie_tv_show_count_by_year(df)
    get_movie_tv_show_percentage(df)
    get_top_countries(df)
    get_content_by_country(df)
    get_content_by_month(df)
    get_wordcloud_from_titles(df)
    get_genre_analysis_movie(df)
    get_genre_analysis_tv_shows(df)


if __name__ == '__main__':
    main()