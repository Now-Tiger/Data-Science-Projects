#!/usr/bin/env/ conda: "base"
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from warnings import filterwarnings
filterwarnings("ignore")


def read_csv(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="ISO-8859-1")
    return data


def remove_columns(column_names: list, data: pd.DataFrame) -> None:
    """ removes columns from the dataset """
    if len(column_names) == 0:
        print("empty columns list")
    elif data.shape[0] == 0:
        print(f"empty dataset: {len(data)}")
        return
    data.drop(columns=column_names, inplace=True)
    return


def get_lowercase_text(column: str, data: pd.DataFrame) -> None:
    """ converts capital/uppercase strings or text into lowercase. """
    if column is None:
        print(f"empty column name")
    if data.shape[0] == 0:
        print(f"empty dataset: {len(data)}")
        return
    data[column] = data[column].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return


def merge_data(data_one: pd.DataFrame, data_two: pd.DataFrame) -> pd.DataFrame:
    """ returns a combined dataset from two datasets.
        Here I'm combining datasets for the product comparison.
        Adding one extra column for own purpose to see the retail price difference 
    """
    
    new_data = data_one.copy()
    new_data['filpk_product'] = data_two['product_name']
    new_data['flipk_retail_price'] = data_two['retail_price']
    new_data['flipk_discounted_price'] = data_two['discounted_price']
    new_data['retail_diff'] = np.where(new_data['amz_retail_price'] == data_two['retail_price'], 
                                       0,
                                       np.absolute(new_data['amz_retail_price'] - data_two['retail_price'])
                                       )
    return new_data


def fetch_product(product_name: str, data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """ returns matching products in the pandas dataframe formate. """
    product_name = product_name.lower()
    if column_name not in data.columns:
        print(f"{column_name} not available in the dataset")
    return data[data[column_name].str.contains(product_name)]



def main() -> None:
    AMZ_PATH = "../data/amz_com_ecommerce_sample.csv"
    FLIPK_PATH = "../data/flipkart_com_ecommerce_sample.csv"

    amazon = read_csv(AMZ_PATH)
    flipkart = read_csv(FLIPK_PATH)
   
    columns = ["uniq_id", "crawl_timestamp", "product_url", "product_category_tree","pid", "image", 
               "is_FK_Advantage_product", "description","product_rating", "overall_rating", "brand", 
               "product_specifications"
               ]

    remove_columns(columns, amazon)
    remove_columns(columns, flipkart)

    get_lowercase_text('product_name', amazon)
    get_lowercase_text('product_name', flipkart)

    amazon.rename(columns={"product_name": "amz_product", 
                           "retail_price": "amz_retail_price", 
                           "discounted_price": "amz_discounted_price"
                           }, 
                  inplace = True
                 )
    combined_data = merge_data(amazon, flipkart)

    print(fetch_product("women's", combined_data, 'amz_product'))
    return


if __name__ == "__main__":
    main()


# -- output --

# (base) D:\>python assesment_two.py
#                                              amz_product  amz_retail_price  amz_discounted_price                                      filpk_product  flipk_retail_price  flipk_discounted_price  retail_diff
# 0                    alisha solid women's cycling shorts               982                   438                alisha solid women's cycling shorts               999.0                   379.0         17.0
# 3                    alisha solid women's cycling shorts               694                   325                alisha solid women's cycling shorts               699.0                   267.0          5.0
# 6                    alisha solid women's cycling shorts              1198                   602                alisha solid women's cycling shorts              1199.0                   479.0          1.0
# 9                    alisha solid women's cycling shorts              1197                   542                alisha solid women's cycling shorts              1199.0                   479.0          2.0
# 11                                carrel printed women's              2281                  1080                             carrel printed women's              2299.0                   910.0         18.0
# ...                                                  ...               ...                   ...                                                ...                 ...                     ...          ...
# 19786                  my addiction women's a-line dress              1293                   934                  my addiction women's a-line dress              1299.0                   779.0          6.0
# 19787                       indibox women's a-line dress              1483                  1136                       indibox women's a-line dress              1499.0                   999.0         16.0
# 19788                         jiiah women's sheath dress              1043                   904                         jiiah women's sheath dress              1049.0                   734.0          6.0
# 19812  lord's red women's slingback peeptoes women heels              1280                   784  lord's red women's slingback peeptoes women heels              1299.0                   650.0         19.0
# 19881  lord's antique gold women's peeptoe heels wome...              1634                  2001  lord's antique gold women's peeptoe heels wome...              1650.0                  1650.0         16.0
# 
# [3789 rows x 7 columns]
