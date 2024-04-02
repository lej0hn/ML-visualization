import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import points
import matplotlib.gridspec as gridspec
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import main


def unit_step_func(u):
    if u<=0:
        return 0
    else:
        return 1
   
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
def draw(X,ax1,ax2,ax3, y_predicted):
    """
    weights (numpy array with shape (3,1)): The weights
    X (numpy array with shape (n,3)): The data
    ax1,ax2,ax3(axes)
    y_predicted(numpy array with size n): The predicted categorization of the data

    Draws the data using scatter and returns the values of the boundry line
    """

    #Draw the plot with the original values and plot everything
    n = X.shape[0]
    
    #Divide into two categories, depending on predicted value
    classA = np.where(y_predicted <= 0)
    classB = np.where(y_predicted > 0)
    
    #First plot
    points.plot_points(X,ax1)
    ax1.set_title('points')
    #Second plot: Draw 0s with 'x' and 1s with 'o'
    ax2.scatter(X[classA, 0], X[classA, 1], marker='x', c='black')
    ax2.scatter(X[classB, 0], X[classB, 1], marker='o', c='magenta')
    #Trito plot: Draw graphma eksodwn protupwn
    ax3.scatter(np.arange(0, n//2), y_predicted[:n//2], marker='x', c='black', label='0')
    ax3.scatter(np.arange(n//2 , n), y_predicted[n//2:], marker='o', c='magenta', label='1')
    #ax3.legend()
    return 

def mse_function(X,y):
    ptp = np.dot(X.T, X)
    pd = np.dot(X.T, y)
    w = np.linalg.inv(ptp).dot(pd)
    return w

class MLPAnimation:
    def __init__(self,  methodos, n_iters=100, lr=0.05, min_error_for_termination = 0.2, hidden_layer_size=3):
        self.methodos = methodos
        self.n_iters = n_iters
        self.hidden_layer_size = hidden_layer_size
        self.lr = lr
        self.weights = None
        self.weights = None
        self.min_error_for_termination = min_error_for_termination

    def fit(self, X, y):
        """
        X(numpy array with shape (n,3)) : Training data
        y(numpy array with size n) : The data categorized in 2 classes

        Trains the model and creates animation

        """
        # Init parameters
        n, n_features = X.shape
        self.weights = np.random.randn(n_features)
        mse = []
        
        # Init plots
        fig,ax1,ax2,ax3,ax4 = create_subplots()
        
        print(self.hidden_layer_size)
        clf = MLPClassifier(hidden_layer_sizes=(self.hidden_layer_size,), max_iter=self.n_iters, alpha=0.0001,
                            solver=self.methodos ,activation='logistic', tol=1e-4, random_state=42,
                            learning_rate_init=self.lr, verbose=False)
        x_line =  np.arange(0, 1, 0.1)
        def update(frame):
            

            clf.partial_fit(X, y, classes=np.unique(y))
            #Clear everything in ax2 and ax3
            ax2.cla()
            ax3.cla()
            ax2.set_title('Training')
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_min, y_max)
            ax3.set_title('Output graph of points')
            
            coef = clf.coefs_
            intercept = clf.intercepts_ 
            y_predicted = clf.predict(X)
            mse.append((1-clf.score(X,y)))
            if self.min_error_for_termination > 1-clf.score(X, y) :
                            ani.event_source.stop()
            draw(X,ax1,ax2,ax3,y_predicted)
            
            #Prospatheia gia emfanish grammwn
            # for i in range(1, coef[0].shape[1]):
            #     y_vals =  -(coef[0][0][i] * x_line - intercept[1][i]) / coef[0][1][i]
            #     ax2.plot(x_line, y_vals, 'k--', linewidth=2)
            # Generate a grid of points within the plot boundaries
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))
            grid_points = np.column_stack((xx.ravel(), yy.ravel()))

            # Predict the class labels for the grid points
            Z = clf.predict(grid_points)
            Z = Z.reshape(xx.shape)

            # Plot the decision boundaries
            ax2.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)


            #Plot 4
            ax4.set_title(f'Mean Squared Error : Epoch - {frame}')
            ax4.plot(mse, '-', color='k')


            self.weights = coef
            return ax2, ax3

            
        ani = animation.FuncAnimation(fig, update, frames=self.n_iters, interval=50, repeat=False)       
        plt.show()
        return self.weights, clf

    def predict(self, X, clf):
        """
        X(numpy array with shape (n,3)) : Testing data
        weights(numpy array ) : The weights produced in the training phase
        
        Returns the algorithms predictions on the testing data
        
        """
        y_predicted = clf.predict(X)
        
        return y_predicted






