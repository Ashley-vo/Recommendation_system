import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def customer_item_matrix(data,customer,item,value):
    matrix = data.pivot_table(index=customer, columns=item, values=value, aggfunc="sum")
    matrix = matrix.applymap(lambda x: 1 if x >0  else 0)
    return matrix


def user_based_matrix(similarityarray,indexseries):
    matrix = pd.DataFrame(similarityarray)
    matrix.columns = indexseries  # set the matrix column names as customerID
    matrix["index_column"] = indexseries # create a column with customer ID
    matrix = matrix.set_index("index_column")  # set index as the customerID
    return matrix


def recommendation_items(dataset,targetcustomer,customer_item_matrix,similariry_matrix):
    # Find the Customer code who has the most similar purchased basket
    similarity_series = similariry_matrix.loc[targetcustomer].sort_values(ascending= False)
    best_match_item = similarity_series.index[1] # the index 0 of the series is the target itself because it has consin  = 1

    item_bought_by_target = set(customer_item_matrix.loc[targetcustomer].iloc[customer_item_matrix.loc[targetcustomer].nonzero()].index)

    item_bought_by_best_match = set(customer_item_matrix.loc[best_match_item].iloc[
                                customer_item_matrix.loc[best_match_item].nonzero()
                            ].index)

    # Find the recommended items to target customer
    item_recommend_to_target = item_bought_by_target - item_bought_by_best_match # extract the StockCode
    item_list = dataset.loc[dataset["StockCode"].isin(item_recommend_to_target),["StockCode","Description"]].drop_duplicates().set_index("StockCode")

    return item_list


if __name__ == '__main__':
    df = pd.read_excel("Online Retail.xlsx", sheet_name="Online Retail")
    # print(df.head())
    # clean data
    ## eliminate return or cancel order
    df = df.loc[df["Quantity"] > 0]
    ## Drop NaN values in CustomerIF
    # print(df.isna().sum())
    df = df.dropna(subset =["CustomerID"])

    # Build customer-to-item matrix
    customer_to_item_matrix = customer_item_matrix(df, "CustomerID", "StockCode", "Quantity")
    #print(customer_to_item_matrix)

    # User-bases approach
    user_user_similarity_array = cosine_similarity(customer_to_item_matrix)
    user_user_similarity_matrix = user_based_matrix(user_user_similarity_array,customer_to_item_matrix.index)

    print(recommendation_items(df,17935,customer_to_item_matrix,user_user_similarity_matrix))


