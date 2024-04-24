from wordcloud import WordCloud
from matplotlib.widgets import Cursor

import matplotlib.pyplot as plt

def visualize_bar_chart(keywords, scores):
    fig, ax = plt.subplots()
    bars = ax.bar(keywords, scores, color='skyblue')

    # Setup Cursor
    cursor = Cursor(ax, useblit=True, color='red', linewidth=2)

    # Initialize an annotation for the cursor, hidden by default.
    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # Update the annotation based on hover
    def update_annot(bar):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_y() + bar.get_height()
        annot.xy = (x, y)
        annot.set_text(f'Score: {bar.get_height():.2f}')
        annot.get_bbox_patch().set_alpha(0.4)

    # Event handler for hovering over the bar chart
    def on_hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for bar in bars:
                cont, _ = bar.contains(event)
                if cont:
                    update_annot(bar)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    # Connect the hover event to the event handler
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    plt.xlabel('Keywords')
    plt.ylabel('TF-IDF Scores')
    plt.title('Top 10 Keywords by TF-IDF in the Top Documents')
    plt.xticks(rotation=45)
    plt.show()


def visualize_word_cloud(keywords, scores):
    # Generate a frequency dictionary for the keywords and scores
    freq_dict = dict(zip(keywords, scores))
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
