import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import points
import matplotlib.gridspec as gridspec
import main


def create_subplots():
    """
    Returns a figure and 3 subplots
    """
    fig = plt.figure(figsize=(10, 10))
    # Create a GridSpec object with 2 rows and 2 columns
    gs = gridspec.GridSpec(2, 2)
    # Define the subplots using the GridSpec object
    ax1 = fig.add_subplot(gs[0, 0])  # Top Left Subplot
    ax2 = fig.add_subplot(gs[0, 1])  # Top Right Subplot
    ax3 = fig.add_subplot(gs[1, 0]) 
    ax4 = fig.add_subplot(gs[1, 1]) 

    return fig,ax1,ax2,ax3,ax4

def draw(X,weights,ax1,ax2,ax3, y_predicted):
    """
    weights (numpy array with shape (3,1)): The weights
    X (numpy array with shape (n,3)): The data
    ax1,ax2,ax3(axes)
    y_predicted(numpy array with size n): The predicted categorization of the data

    Draws the data using scatter and returns the values of the boundry line
    """

    #Draw the plot with the original values and plot everything
    n = X.shape[0]
    # Update the decision boundary line
    x_line = X[:,0]
    y_line = -(weights[0]*x_line - weights[2])/weights[1]
    #Divide into two categories, depending on predicted value
    classA = np.where(y_predicted <= 0)
    classB = np.where(y_predicted > 0)
    
    #Prwto plot
    points.plot_protupa(X,ax1)
    #Deutero plot: Draw 0s with 'x' and 1s with 'o'
    ax2.scatter(X[classA, 0], X[classA, 1], marker='x', c='black')
    ax2.scatter(X[classB, 0], X[classB, 1], marker='o', c='magenta')
    #Trito plot: Draw graphma eksodwn protupwn
    ax3.scatter(np.arange(0, n//2), y_predicted[:n//2], marker='x', c='black', label='0')
    ax3.scatter(np.arange(n//2 , n), y_predicted[n//2:], marker='o', c='magenta', label='1')
    #ax3.legend()
    return x_line, y_line


def mse_function(X,y):
    ptp = np.dot(X.T, X)
    pd = np.dot(X.T, y)
    w = np.linalg.inv(ptp).dot(pd)
    return w

class AdalineAnimation:
    def __init__(self, n_iters=100, lr=0.05, min_sfalma_for_termination = 0.2):
        self.n_iters = n_iters
        self.lr = lr
        self.weights = None
        self.min_sfalma_for_termination = min_sfalma_for_termination
    
    def fit(self, X, y):
        """
        X(numpy array with shape (n,3)) : Training data
        y(numpy array with size n) : The data categorized in 2 classes

        Trains the model and creates animation

        """
        # Init parameters
        n, n_features = X.shape
        y_predicted = np.ones(n)
        self.weights = np.random.randn(n_features)
        delta = np.zeros(n)
        mse = []
        w = mse_function(X,y)
        x_line = X[:,0]
        y_line = -(w[0]*x_line - w[2])/w[1]
        
        #Init subplots
        fig,ax1,ax2,ax3,ax4 = create_subplots()
        ax1.set_title('Prwtupa me grammh mean square error')
        
        line_mse = ax1.plot(x_line,y_line, 'r-', lw=2)
        
        

        def update(frame):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) 
                y_predicted[idx] = linear_output
                delta[idx] = y[idx] - y_predicted[idx]
                # Adaline weight update rule
                self.weights += self.lr * delta[idx] * x_i
            sfalma=0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) 
                y_predicted[idx] = linear_output
                delta[idx] = y[idx] - y_predicted[idx]
                sfalma += delta[idx]**2
            if self.min_sfalma_for_termination > sfalma/n :
                ani.event_source.stop()
            mse.append(sfalma/n)

            #Clear everything in ax2 and ax3
            ax2.cla()
            ax3.cla()
            line, = ax2.plot([], [], 'r-', lw=2)
            #Ta 3 prwta plot
            ax2.set_title('Training me grammh adeline')
            ax3.set_title('Graphima eksodwn protupwn')
            x_line,y_line = draw(X,self.weights,ax1,ax2,ax3,y_predicted)
            line.set_data(x_line,y_line)
            #Plot 4
            ax4.set_title(f'Mean Squared Error : Epoxh - {frame}')
            ax4.plot(mse, '-', color='k')

            return line, ax2, ax3

        ani = animation.FuncAnimation(fig, update, frames=self.n_iters, interval=200, repeat=False)       
        plt.show()
        return self.weights, w
    
    def fit_radial(self, X, y):
        """
        X(numpy array with shape (n,3)) : Training data
        y(numpy array with size n) : The data categorized in 2 classes

        Trains the model and creates animation

        """
        # Init parameters
        n, n_features = X.shape
        y_predicted = np.ones(n)
        self.weights = np.random.randn(n_features)
        delta = np.zeros(n)
        mse = []
        w = mse_function(X,y)
     
        #Init subplots
        fig,ax1,ax2,ax3,ax4 = create_subplots()
        ax1.set_title('Prwtupa ')
        

        def update(frame):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) 
                y_predicted[idx] = linear_output
                delta[idx] = y[idx] - y_predicted[idx]
                # Adaline weight update rule
                self.weights += self.lr * delta[idx] * x_i
            sfalma=0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) 
                y_predicted[idx] = linear_output
                delta[idx] = y[idx] - y_predicted[idx]
                sfalma += delta[idx]**2
            if self.min_sfalma_for_termination > sfalma/n :
                ani.event_source.stop()
            mse.append(sfalma/n)

            #Clear everything in ax2 and ax3
            ax2.cla()
            ax3.cla()
            line, = ax2.plot([], [], 'r-', lw=2)
            #Ta 3 prwta plot
            ax2.set_title('Training me grammh adeline')
            ax3.set_title('Graphima eksodwn protupwn')
            x_line,y_line = draw(X,self.weights,ax1,ax2,ax3,y_predicted)
            line.set_data(x_line,y_line)

            #Plot 4
            ax4.set_title(f'Mean Squared Error : Epoxh - {frame}')
            ax4.plot(mse, '-', color='k')

            return line, ax2, ax3

        ani = animation.FuncAnimation(fig, update, frames=self.n_iters, interval=200, repeat=False)       
        plt.show()
        return self.weights, w

    def predict(self, X, weights):
        """
        X(numpy array with shape (n,3)) : Testing data
        weights(numpy array ) : The weights produced in the training phase
        
        Returns the algorithms predictions on the testing data
        
        """

        y_predicted = np.ones(X.shape[0])
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, weights) 
            y_predicted[idx] = linear_output
        return y_predicted   

    





