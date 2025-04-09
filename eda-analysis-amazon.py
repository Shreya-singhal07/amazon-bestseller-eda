# Exploratory Data Analysis on Amazon Bestsellers
# -------------------------------------------------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plot aesthetics
plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("### Amazon Bestsellers - Exploratory Data Analysis ###")
print("-" * 55)

# Load the dataset
# For this example, we'll use a common Amazon Books dataset available online
# In a real-world scenario, you might need to scrape this data or use an API

# Simulating data download
print("Loading Amazon bestsellers data...")

# Create sample data - in a real project, you would load actual data
def create_sample_bestsellers_data(n=200):
    """Create a sample dataset of Amazon bestsellers"""
    np.random.seed(42)
    
    # Define possible categories and authors
    categories = ['Fiction', 'Non-Fiction', 'Mystery', 'Science Fiction', 'Romance', 
                 'Biography', 'Self-Help', 'Business', 'Children', 'Young Adult']
    
    authors = ['J.K. Rowling', 'Stephen King', 'Malcolm Gladwell', 'James Patterson',
              'Brené Brown', 'Michelle Obama', 'J.R.R. Tolkien', 'Dale Carnegie',
              'Yuval Noah Harari', 'Robert Greene', 'Paulo Coelho', 'Rick Riordan',
              'George R.R. Martin', 'Rupi Kaur', 'John Grisham', 'Colleen Hoover']
    
    publishers = ['Penguin Random House', 'HarperCollins', 'Simon & Schuster', 
                 'Hachette Book Group', 'Macmillan Publishers', 'Scholastic',
                 'Wiley', 'Oxford University Press', 'Bloomsbury']
    
    # Generate data
    data = {
        'Title': [f"Book Title {i}" for i in range(1, n+1)],
        'Author': np.random.choice(authors, n),
        'Price': np.round(np.random.uniform(7.99, 29.99, n), 2),
        'Rating': np.round(np.random.uniform(1, 5, n), 1),
        'Reviews': np.random.randint(5, 10000, n),
        'Category': np.random.choice(categories, n),
        'Publisher': np.random.choice(publishers, n),
        'Pages': np.random.randint(100, 900, n),
        'Year': np.random.randint(2010, 2025, n),
        'Bestseller_Rank': np.random.randint(1, 1000, n),
        'Format': np.random.choice(['Paperback', 'Hardcover', 'Kindle', 'Audiobook'], n),
        'Language': np.random.choice(['English', 'Spanish', 'French', 'German'], n, p=[0.85, 0.07, 0.05, 0.03])
    }
    
    # Add some bestseller series
    popular_series = ['Harry Potter', 'Game of Thrones', 'Lord of the Rings', 
                      'Hunger Games', 'Percy Jackson']
    
    # Replace some titles with series books
    for series in popular_series:
        series_indices = np.random.choice(range(n), size=5, replace=False)
        for i, idx in enumerate(series_indices):
            data['Title'][idx] = f"{series} {i+1}"
    
    df = pd.DataFrame(data)
    
    # Create some correlations (price vs pages, rating vs reviews)
    df['Price'] = df['Price'] + df['Pages'] * 0.015
    df['Rating'] = df['Rating'] + df['Reviews'] * 0.0001
    df['Rating'] = df['Rating'].apply(lambda x: min(5, x))
    
    # Add some seasonality: books from recent years tend to cost more
    df['Price'] += (df['Year'] - 2010) * 0.3
    
    # Make certain categories more expensive
    category_price_boost = {
        'Business': 5.0,
        'Non-Fiction': 3.0,
        'Biography': 2.0
    }
    
    for cat, boost in category_price_boost.items():
        df.loc[df['Category'] == cat, 'Price'] += boost
    
    # Make books with lots of reviews have generally higher ratings
    df.loc[df['Reviews'] > 5000, 'Rating'] += 0.5
    df['Rating'] = df['Rating'].apply(lambda x: min(5, x))
    
    # Add a column for bestseller flag (top 100)
    df['Bestseller'] = df['Bestseller_Rank'] <= 100
    
    return df

# Create and display the dataset
df_books = create_sample_bestsellers_data(200)
print(f"Data loaded successfully: {df_books.shape[0]} books")
print("-" * 55)

# Display the first few rows
print("\nFirst few entries in the dataset:")
print(df_books.head())

# Basic data exploration
print("\n### Data Overview ###")
print("\nDataset shape:", df_books.shape)
print("\nData types:")
print(df_books.dtypes)

print("\nBasic statistics:")
print(df_books.describe())

print("\nMissing values:")
print(df_books.isnull().sum())

# Add a publication decade column
df_books['Decade'] = (df_books['Year'] // 10) * 10
df_books['Decade'] = df_books['Decade'].astype(str) + 's'

# Distribution Analysis
print("\n### Distribution Analysis ###")

# Price Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df_books['Price'], kde=True, bins=30)
plt.title('Distribution of Book Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.axvline(df_books['Price'].mean(), color='red', linestyle='--', label=f'Mean: ${df_books["Price"].mean():.2f}')
plt.axvline(df_books['Price'].median(), color='green', linestyle='--', label=f'Median: ${df_books["Price"].median():.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.close()

print("Price statistics:")
print(f"Mean price: ${df_books['Price'].mean():.2f}")
print(f"Median price: ${df_books['Price'].median():.2f}")
print(f"Min price: ${df_books['Price'].min():.2f}")
print(f"Max price: ${df_books['Price'].max():.2f}")

# Rating Distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Rating', data=df_books, order=sorted(df_books['Rating'].unique()))
plt.title('Distribution of Book Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('rating_distribution.png')
plt.close()

print("\nRating statistics:")
print(f"Mean rating: {df_books['Rating'].mean():.2f}")
print(f"Median rating: {df_books['Rating'].median():.2f}")
print(f"Most common rating: {df_books['Rating'].mode()[0]}")

# Category Distribution
plt.figure(figsize=(14, 8))
category_counts = df_books['Category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Distribution of Book Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('category_distribution.png')
plt.close()

print("\nCategory distribution:")
for category, count in category_counts.items():
    print(f"{category}: {count} books ({count/len(df_books)*100:.1f}%)")

# Format Distribution
plt.figure(figsize=(10, 6))
format_counts = df_books['Format'].value_counts()
plt.pie(format_counts, labels=format_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Book Format Distribution')
plt.axis('equal')  
plt.tight_layout()
plt.savefig('format_distribution.png')
plt.close()

print("\nFormat distribution:")
for format_type, count in format_counts.items():
    print(f"{format_type}: {count} books ({count/len(df_books)*100:.1f}%)")

# Publication Year Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df_books['Year'], kde=True, bins=15, discrete=True)
plt.title('Distribution of Publication Years')
plt.xlabel('Year')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('year_distribution.png')
plt.close()

print("\nPublication year statistics:")
print(f"Oldest book: {df_books['Year'].min()}")
print(f"Newest book: {df_books['Year'].max()}")
print(f"Most common publication year: {df_books['Year'].mode()[0]}")

# Relationship Analysis
print("\n### Relationship Analysis ###")

# Price vs. Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='Rating', data=df_books, alpha=0.6)
plt.title('Relationship Between Price and Rating')
plt.xlabel('Price ($)')
plt.ylabel('Rating')
plt.grid(True)
plt.tight_layout()
plt.savefig('price_vs_rating.png')
plt.close()

correlation = df_books['Price'].corr(df_books['Rating'])
print(f"Correlation between Price and Rating: {correlation:.2f}")

# Pages vs. Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pages', y='Price', data=df_books, alpha=0.6)
plt.title('Relationship Between Number of Pages and Price')
plt.xlabel('Number of Pages')
plt.ylabel('Price ($)')
plt.grid(True)
plt.tight_layout()
plt.savefig('pages_vs_price.png')
plt.close()

correlation = df_books['Pages'].corr(df_books['Price'])
print(f"Correlation between Pages and Price: {correlation:.2f}")

# Reviews vs. Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Reviews', y='Rating', data=df_books, alpha=0.6)
plt.title('Relationship Between Number of Reviews and Rating')
plt.xlabel('Number of Reviews')
plt.ylabel('Rating')
plt.xscale('log')  # Log scale for better visualization
plt.grid(True)
plt.tight_layout()
plt.savefig('reviews_vs_rating.png')
plt.close()

correlation = df_books['Reviews'].corr(df_books['Rating'])
print(f"Correlation between Reviews and Rating: {correlation:.2f}")

# Price by Category
plt.figure(figsize=(14, 8))
sns.boxplot(x='Category', y='Price', data=df_books)
plt.title('Price Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Price ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('price_by_category.png')
plt.close()

print("\nAverage Price by Category:")
category_price = df_books.groupby('Category')['Price'].mean().sort_values(ascending=False)
for category, avg_price in category_price.items():
    print(f"{category}: ${avg_price:.2f}")

# Rating by Category
plt.figure(figsize=(14, 8))
sns.boxplot(x='Category', y='Rating', data=df_books)
plt.title('Rating Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('rating_by_category.png')
plt.close()

print("\nAverage Rating by Category:")
category_rating = df_books.groupby('Category')['Rating'].mean().sort_values(ascending=False)
for category, avg_rating in category_rating.items():
    print(f"{category}: {avg_rating:.2f}")

# Price by Format
plt.figure(figsize=(12, 6))
sns.boxplot(x='Format', y='Price', data=df_books)
plt.title('Price Distribution by Format')
plt.xlabel('Format')
plt.ylabel('Price ($)')
plt.tight_layout()
plt.savefig('price_by_format.png')
plt.close()

print("\nAverage Price by Format:")
format_price = df_books.groupby('Format')['Price'].mean().sort_values(ascending=False)
for format_type, avg_price in format_price.items():
    print(f"{format_type}: ${avg_price:.2f}")

# Rating by Format
plt.figure(figsize=(12, 6))
sns.boxplot(x='Format', y='Rating', data=df_books)
plt.title('Rating Distribution by Format')
plt.xlabel('Format')
plt.ylabel('Rating')
plt.tight_layout()
plt.savefig('rating_by_format.png')
plt.close()

print("\nAverage Rating by Format:")
format_rating = df_books.groupby('Format')['Rating'].mean().sort_values(ascending=False)
for format_type, avg_rating in format_rating.items():
    print(f"{format_type}: {avg_rating:.2f}")

# Author Analysis
print("\n### Author Analysis ###")

# Top 10 Authors by Book Count
top_authors = df_books['Author'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_authors.index, y=top_authors.values)
plt.title('Top 10 Authors by Number of Books')
plt.xlabel('Author')
plt.ylabel('Number of Books')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_authors.png')
plt.close()

print("Top 10 Authors by Book Count:")
for author, count in top_authors.items():
    print(f"{author}: {count} books")

# Average Rating by Author (for authors with at least 3 books)
author_ratings = df_books.groupby('Author').filter(lambda x: len(x) >= 3).groupby('Author')['Rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=author_ratings.head(10).index, y=author_ratings.head(10).values)
plt.title('Top 10 Authors by Average Rating (Min. 3 Books)')
plt.xlabel('Author')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_rated_authors.png')
plt.close()

print("\nTop 10 Authors by Average Rating (Min. 3 Books):")
for author, rating in author_ratings.head(10).items():
    print(f"{author}: {rating:.2f}")

# Temporal Analysis
print("\n### Temporal Analysis ###")

# Average Price Trend Over Years
yearly_price = df_books.groupby('Year')['Price'].mean()
plt.figure(figsize=(12, 6))
sns.lineplot(x=yearly_price.index, y=yearly_price.values)
plt.title('Average Book Price Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Price ($)')
plt.grid(True)
plt.tight_layout()
plt.savefig('price_trend.png')
plt.close()

print("Price trend over time:")
start_year = df_books['Year'].min()
end_year = df_books['Year'].max()
start_price = yearly_price[start_year]
end_price = yearly_price[end_year]
price_change = ((end_price - start_price) / start_price) * 100
print(f"From {start_year} to {end_year}, average book price changed from ${start_price:.2f} to ${end_price:.2f}")
print(f"That's a {price_change:.1f}% {'increase' if price_change > 0 else 'decrease'}")

# Average Rating Trend Over Years
yearly_rating = df_books.groupby('Year')['Rating'].mean()
plt.figure(figsize=(12, 6))
sns.lineplot(x=yearly_rating.index, y=yearly_rating.values)
plt.title('Average Book Rating Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.grid(True)
plt.tight_layout()
plt.savefig('rating_trend.png')
plt.close()

print("\nRating trend over time:")
start_rating = yearly_rating[start_year]
end_rating = yearly_rating[end_year]
rating_change = ((end_rating - start_rating) / start_rating) * 100
print(f"From {start_year} to {end_year}, average book rating changed from {start_rating:.2f} to {end_rating:.2f}")
print(f"That's a {rating_change:.1f}% {'increase' if rating_change > 0 else 'decrease'}")

# Category Popularity Over Decades
decade_category = pd.crosstab(df_books['Decade'], df_books['Category'])
decade_category_pct = decade_category.div(decade_category.sum(axis=1), axis=0) * 100

plt.figure(figsize=(14, 10))
decade_category_pct.plot(kind='bar', stacked=True)
plt.title('Category Popularity by Decade')
plt.xlabel('Decade')
plt.ylabel('Percentage (%)')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('category_by_decade.png')
plt.close()

print("\nMost popular category by decade:")
for decade in sorted(df_books['Decade'].unique()):
    top_category = decade_category.loc[decade].idxmax()
    print(f"{decade}: {top_category} ({decade_category.loc[decade, top_category]} books)")

# Format Popularity Over Decades
decade_format = pd.crosstab(df_books['Decade'], df_books['Format'])
decade_format_pct = decade_format.div(decade_format.sum(axis=1), axis=0) * 100

plt.figure(figsize=(12, 8))
decade_format_pct.plot(kind='bar', stacked=True)
plt.title('Format Popularity by Decade')
plt.xlabel('Decade')
plt.ylabel('Percentage (%)')
plt.legend(title='Format', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('format_by_decade.png')
plt.close()

print("\nMost popular format by decade:")
for decade in sorted(df_books['Decade'].unique()):
    top_format = decade_format.loc[decade].idxmax()
    print(f"{decade}: {top_format} ({decade_format.loc[decade, top_format]} books)")

# Bestseller Analysis
print("\n### Bestseller Analysis ###")

# Bestseller Count by Category
bestseller_category = df_books[df_books['Bestseller']]['Category'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=bestseller_category.index, y=bestseller_category.values)
plt.title('Number of Bestsellers by Category')
plt.xlabel('Category')
plt.ylabel('Number of Bestsellers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('bestsellers_by_category.png')
plt.close()

print("Bestsellers by category:")
total_bestsellers = df_books['Bestseller'].sum()
for category, count in bestseller_category.items():
    print(f"{category}: {count} bestsellers ({count/total_bestsellers*100:.1f}% of all bestsellers)")

# Bestseller Rate by Category
category_bestseller_rate = df_books.groupby('Category')['Bestseller'].mean() * 100
plt.figure(figsize=(12, 6))
sns.barplot(x=category_bestseller_rate.index, y=category_bestseller_rate.values)
plt.title('Bestseller Rate by Category (%)')
plt.xlabel('Category')
plt.ylabel('Bestseller Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('bestseller_rate_by_category.png')
plt.close()

print("\nBestseller rate by category:")
for category, rate in category_bestseller_rate.sort_values(ascending=False).items():
    print(f"{category}: {rate:.1f}% of books are bestsellers")

# Average Price: Bestsellers vs Non-Bestsellers
plt.figure(figsize=(10, 6))
sns.boxplot(x='Bestseller', y='Price', data=df_books)
plt.title('Price Distribution: Bestsellers vs Non-Bestsellers')
plt.xlabel('Bestseller')
plt.ylabel('Price ($)')
plt.tight_layout()
plt.savefig('bestseller_price.png')
plt.close()

bestseller_avg_price = df_books[df_books['Bestseller']]['Price'].mean()
nonbestseller_avg_price = df_books[~df_books['Bestseller']]['Price'].mean()
print(f"\nAverage price of bestsellers: ${bestseller_avg_price:.2f}")
print(f"Average price of non-bestsellers: ${nonbestseller_avg_price:.2f}")
price_diff_pct = ((bestseller_avg_price - nonbestseller_avg_price) / nonbestseller_avg_price) * 100
print(f"Bestsellers are {abs(price_diff_pct):.1f}% {'more' if price_diff_pct > 0 else 'less'} expensive on average")

# Average Rating: Bestsellers vs Non-Bestsellers
plt.figure(figsize=(10, 6))
sns.boxplot(x='Bestseller', y='Rating', data=df_books)
plt.title('Rating Distribution: Bestsellers vs Non-Bestsellers')
plt.xlabel('Bestseller')
plt.ylabel('Rating')
plt.tight_layout()
plt.savefig('bestseller_rating.png')
plt.close()

bestseller_avg_rating = df_books[df_books['Bestseller']]['Rating'].mean()
nonbestseller_avg_rating = df_books[~df_books['Bestseller']]['Rating'].mean()
print(f"\nAverage rating of bestsellers: {bestseller_avg_rating:.2f}")
print(f"Average rating of non-bestsellers: {nonbestseller_avg_rating:.2f}")
rating_diff = bestseller_avg_rating - nonbestseller_avg_rating
print(f"Bestsellers are rated {abs(rating_diff):.2f} points {'higher' if rating_diff > 0 else 'lower'} on average")

# Publisher Analysis
print("\n### Publisher Analysis ###")

# Top Publishers by Bestseller Count
publisher_bestsellers = df_books[df_books['Bestseller']].groupby('Publisher').size().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=publisher_bestsellers.head(10).index, y=publisher_bestsellers.head(10).values)
plt.title('Top 10 Publishers by Number of Bestsellers')
plt.xlabel('Publisher')
plt.ylabel('Number of Bestsellers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('publishers_by_bestsellers.png')
plt.close()

print("Top publishers by bestseller count:")
for publisher, count in publisher_bestsellers.head(10).items():
    print(f"{publisher}: {count} bestsellers")

# Publisher Bestseller Rate
publisher_counts = df_books.groupby('Publisher').size()
publisher_bestseller_rate = df_books.groupby('Publisher')['Bestseller'].mean() * 100

# Filter publishers with at least 5 books
major_publishers = publisher_counts[publisher_counts >= 5].index
major_publisher_rates = publisher_bestseller_rate.loc[major_publishers].sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=major_publisher_rates.head(10).index, y=major_publisher_rates.head(10).values)
plt.title('Top 10 Publishers by Bestseller Rate (Min. 5 Books)')
plt.xlabel('Publisher')
plt.ylabel('Bestseller Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('publisher_bestseller_rate.png')
plt.close()

print("\nTop publishers by bestseller rate (min. 5 books):")
for publisher, rate in major_publisher_rates.head(10).items():
    book_count = publisher_counts[publisher]
    print(f"{publisher}: {rate:.1f}% bestseller rate ({book_count} total books)")

# Correlation Analysis
print("\n### Correlation Analysis ###")

# Select numeric columns
numeric_cols = ['Price', 'Rating', 'Reviews', 'Pages', 'Year', 'Bestseller_Rank']
corr_matrix = df_books[numeric_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

print("Strong correlations found:")
# Extract the upper triangle of the correlation matrix
upper_triangle = corr_matrix.where(mask == False)
# Find strong correlations (absolute value > 0.3)
strong_corrs = [(i, j, corr_matrix.loc[i, j]) 
                for i in corr_matrix.index 
                for j in corr_matrix.columns 
                if abs(corr_matrix.loc[i, j]) > 0.3 and i != j and not pd.isna(upper_triangle.loc[i, j])]

for var1, var2, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
    print(f"{var1} and {var2}: {corr:.2f}")

# Summary Statistics
print("\n### Summary Statistics ###")

print("Overall dataset summary:")
print(f"- Total number of books: {len(df_books)}")
print(f"- Number of bestsellers: {df_books['Bestseller'].sum()} ({df_books['Bestseller'].mean()*100:.1f}%)")
print(f"- Number of authors: {df_books['Author'].nunique()}")
print(f"- Number of publishers: {df_books['Publisher'].nunique()}")
print(f"- Average book price: ${df_books['Price'].mean():.2f}")
print(f"- Average book rating: {df_books['Rating'].mean():.2f}")
print(f"- Most common category: {df_books['Category'].mode()[0]}")
print(f"- Most common format: {df_books['Format'].mode()[0]}")

print("\n### Key Insights ###")
print("\n1. Price Insights:")
print(f"   - Most expensive category: {category_price.index[0]} (${category_price.iloc[0]:.2f})")
print(f"   - Least expensive category: {category_price.index[-1]} (${category_price.iloc[-1]:.2f})")
print(f"   - Most expensive format: {format_price.index[0]} (${format_price.iloc[0]:.2f})")
print(f"   - Price trend: {'Increasing' if price_change > 0 else 'Decreasing'} over time ({abs(price_change):.1f}%)")

print("\n2. Rating Insights:")
print(f"   - Highest rated category: {category_rating.index[0]} ({category_rating.iloc[0]:.2f})")
print(f"   - Lowest rated category: {category_rating.index[-1]} ({category_rating.iloc[-1]:.2f})")
print(f"   - Highest rated format: {format_rating.index[0]} ({format_rating.iloc[0]:.2f})")
print(f"   - Rating trend: {'Increasing' if rating_change > 0 else 'Decreasing'} over time ({abs(rating_change):.1f}%)")

print("\n3. Bestseller Insights:")
print(f"   - Category with most bestsellers: {bestseller_category.index[0]} ({bestseller_category.iloc[0]})")
print(f"   - Category with highest bestseller rate: {category_bestseller_rate.idxmax()} ({category_bestseller_rate.max():.1f}%)")
print(f"   - Publisher with most bestsellers: {publisher_bestsellers.index[0]} ({publisher_bestsellers.iloc[0]})")
print(f"   - Bestsellers vs non-bestsellers price difference: {abs(price_diff_pct):.1f}% {'higher' if price_diff_pct > 0 else 'lower'}")
print(f"   - Bestsellers vs non-bestsellers rating difference: {abs(rating_diff):.2f} points {'higher' if rating_diff > 0 else 'lower'}")

print("\n4. Format and Category Trends:")
print(f"   - Most popular category in the earliest decade: {decade_category.loc[sorted(df_books['Decade'].unique())[0]].idxmax()}")
print(f"   - Most popular category in the latest decade: {decade_category.loc[sorted(df_books['Decade'].unique())[-1]].idxmax()}")
print(f"   - Most popular format in the earliest decade: {decade_format.loc[sorted(df_books['Decade'].unique())[0]].idxmax()}")
print(f"   - Most popular format in the latest decade: {decade_format.loc[sorted(df_books['Decade'].unique())[-1]].idxmax()}")

print("\n5. Author Insights:")
print(f"   - Author with most books: {top_authors.index[0]} ({top_authors.iloc[0]} books)")
print(f"   - Highest rated author (min. 3 books): {author_ratings.index[0]} ({author_ratings.iloc[0]:.2f})")

print("\n### Conclusion ###")

