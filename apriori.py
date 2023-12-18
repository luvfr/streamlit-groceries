import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

# Load Dataset
df = pd.read_csv("Groceries_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

df["month"] = df['Date'].dt.month
df["day"] = df['Date'].dt.weekday

df["month"].replace([i for i in range(1, 12 + 1)], ["January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December"], inplace=True)
df["day"].replace([i for i in range(6 + 1)], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], inplace=True)

st.title("Market Basket Analysis Menggunakan Algoritma Apriori")

def get_data(month = '', day = ''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Result!"
5
def user_input_features():
    item = st.selectbox("itemDescription", ['abrasive cleaner', 'artif. sweetener', 'baby cosmetics', 'bags', 'baking powder',
        'bathroom cleaner', 'beef', 'berries', 'beverages', 'bottled beer', 'bottled water', 'butter', 'butter milk',
        'cake bar', 'candles', 'candy', 'canned beer', 'canned fish', 'canned fruit', 'canned vegetables', 'cat food', 'cereals', 'chewing gum', 'chicken', 'chocolate',
        'chocolate marshmallow', 'citrus fruit', 'cleaner', 'cling film/bags', 'cocoa drinks', 'coffee', 'condensed milk', 'cooking chocolate', 'cookware',
        'cream', 'cream cheese', 'curd', 'curd cheese', 'decalcifier', 'dental care', 'dessert', 'detergent', 'dish cleaner', 'dishes', 'dog food', 'domestic eggs', 'female sanitary eggs', 'finished product'
        'fish', 'flour', 'flower (seeds)', 'flower soil/fertilizer', 'frankfurter', 'frozen chicken', 'frozen dessert', 'frozen fish', 'frozen fruits', 'frozen meals', 'frozen potato products', 'frozen vegetables', 
        'fruit/vegetable juice', 'grapes', 'hair spray', 'ham', 'hamburger meat', 'hard cheese', 'herbs', 'honey', 'house keeping products', 'hygiene articles', 'ice cream', 'instant coffee', 'instant food products', 
        'jam', 'ketchup', 'kitchen towels', 'kitchen utensil', 'light bulbs', 'liqueur', 'liquor', 'liquor (appetizer)', 'liver loaf', 'long life bakery product', 'make up remover', 'male cosmetics', 'margarine', 
        'mayonnaise', 'meat', 'meat spreads', 'misc. beverages', 'mustard', 'napkins', 'newspapers', 'nut snack', 'nuts/prunes', 'oil', 'onions', 'organic products', 'organic sausage', 'other vegetables', 
        'packaged fruit/vegetables', 'pasta', 'pastry', 'pet care', 'photo/film', 'pickled/film', 'pickled vegetables', 'pip fruit', 'popcorn', 'pork', 'pot plants', 'potato products', 'preservation products', 
        'processed cheese', 'prosecco', 'pudding powder', 'ready soups', 'red/blush wine', 'rice', 'roll products', 'rolls/buns', 'root vegetables', 'rubbing alcohol', 'rum', 'salad dressing', 'salt', 'salty snack', 
        'sauces', 'sausage', 'seasonal products', 'semi-finished bread', 'shopping bags', 'skin care', 'sliced cheese', 'snack products', 'soap', 'soda', 'soft cheese', 'softener', 'soups', 'sparkling wine', 'specialty bar', 
        'specialty cheese', 'specialty chocolate', 'specialty fat', 'specialty vegetables', 'spices', 'spread cheeses', 'sugar', 'sweet spreads', 'syrup', 'tea', 'tidbits', 'toilet cleaner' , 'tropical fruit', 'turkey', 'UHT-milk', 
        'vinegar', 'waffles', 'whipped/sour cream', 'whisky', 'white bread', 'white wine', 'whole milk', 'yogurt', 'zwieback'])
    month = st.select_slider("Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    day = st.select_slider("Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], value = "Saturday")
    
    return month, day, item

month, day, item = user_input_features()

data = get_data(month, day)

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    
if type(data) != type ("No Result!"):
    item_count = data.groupby(["Member_number", "itemDescription"])["itemDescription"].count().reset_index(name = "Count")
    item_count_pivot = item_count.pivot_table(index = 'Member_number', columns = 'itemDescription', values = 'Count', aggfunc = 'sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)
    
    support = 0.01
    frequency_items = apriori(item_count_pivot, min_support = support, use_colnames = True)
    
    metric = "lift"
    min_threshold = 1
    
    rules = association_rules(frequency_items, metric = metric, min_threshold = min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)
    
def parse_list(x):
    x = list(x)
    if len(x) ==  1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)
    
def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()
    
    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    return list(data.loc[data["antecedents"] == item_antecedents].iloc[0,:])
    
if type(data) != type("No Result!"):
    st.markdown("Hasil Rekomendasi : ")
    st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{return_item_df(item)[1]}** secara bersamaan")
else:
    st.error("Tidak ada hasil yang ditemukan")
    

