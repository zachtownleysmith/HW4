from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Create Variable for running the script
run = True

while run:
    service = input('Input analysis to perform (regression or cluster)....')
    print('')

    if service == 'regression':
        boston_X, boston_y = load_boston(return_X_y=True)
        data = load_boston()

        # Create an Instance of the Linear Regression Class and fit Y vs X for the 13 Boston Housing inputs
        lin_regression = LinearRegression()
        lin_regression.fit(boston_X, boston_y)

        # Compute the range of each X Variable.
        # The variable with the highest impact will have the largest range * regression coefficient product
        boston_X_rng = np.max(boston_X, axis=0) - np.min(boston_X, axis=0)
        price_effect = boston_X_rng * lin_regression.coef_

        # Determine the location of largest (positive or negative) impact on housing price
        max_index = np.argmax(abs(price_effect))

        # Result Reporting
        print("The feature most impacting house prices was: " + data.feature_names[max_index])
        print("The range of this variable in the Boston Dataset was: " + str(round(boston_X_rng[max_index], 4)))
        print("The price change per unit X for this variable was : " + str(round(lin_regression.coef_[max_index], 4)))
        print('')

    elif service == 'cluster':
        wine_data = load_wine()

        # Define function for returning Sum of Squared Errors for KMeans fit
        def fit_kmeans(x):
            cluster_model = KMeans(n_clusters=x)
            fit_model = cluster_model.fit(wine_data['data'])
            sum_error2 = fit_model.inertia_
            return sum_error2

        # Define Placeholder DataFrame for Results
        df = pd.DataFrame(columns=['K', 'SSE'])

        # Loop through clusters from 1 - 15
        for i in range(1, 16):
            error_sum = fit_kmeans(x=i)
            temp = {'K': i, 'SSE': error_sum}
            df = df.append(temp, ignore_index=True)

        # Results Plotting
        plt.plot(df['K'], df['SSE'], color='r')
        plt.title('KMeans Elbow Plot for Sklearn Wine Dataset')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Sum of Squared Errors')

        print("By visual inspection, using the Elbow Heuristic we confirm that the data has three classes")
        print('Close the plot to continue the code!')
        print('')
        plt.show()

    else:
        print('Requested Service Not Recognized')
        print('')

    run_again = input('Run another Analysis? (Y/N)...')
    if run_again == 'Y':
        run = True
    else:
        run = False




