import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import points
import matplotlib.gridspec as gridspec
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
    ax3 = fig.add_subplot(gs[1, :]) 
    return fig,gs,ax1,ax2,ax3

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
    classA = np.where(y_predicted == 0)
    classB = np.where(y_predicted == 1)
    
    #Prwto plot
    points.plot_protupa(X,ax1)
    ax1.set_title('Protupa')
    #Deutero plot: Draw 0s with 'x' and 1s with 'o'
    ax2.scatter(X[classA, 0], X[classA, 1], marker='x', c='black')
    ax2.scatter(X[classB, 0], X[classB, 1], marker='o', c='magenta')
    #Trito plot: Draw graphma eksodwn protupwn
    ax3.scatter(np.arange(0, n//2), y_predicted[:n//2], marker='x', c='black', label='0')
    ax3.scatter(np.arange(n//2 , n), y_predicted[n//2:], marker='o', c='magenta', label='1')
    ax3.legend()
    return x_line, y_line

class PerceptronAnimation:
    def __init__(self, n_iters=100, lr=0.05):
        self.n_iters = n_iters
        self.lr = lr
        self.activation_func = unit_step_func
        self.weights = None
    
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
        
        # Init plots
        fig,gs,ax1,ax2,ax3 = create_subplots()
        
        
        
        def update(frame): 

            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) 
                y_predicted[idx] = self.activation_func(linear_output)

                # Perceptron weight update rule
                if (y[idx] != y_predicted[idx]):
                    self.weights += self.lr * (y[idx] - y_predicted[idx]) * x_i
                    
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) 
                y_predicted[idx] = self.activation_func(linear_output)
            #Clear everything in ax2 and ax3
            ax2.cla()
            ax3.cla()
            ax2.set_title('Training')
            ax3.set_title('Graphima eksodwn protupwn')
            
            line, = ax2.plot([], [], 'r-', lw=2)
            x_line,y_line = draw(X,self.weights,ax1,ax2,ax3,y_predicted)
            line.set_data(x_line,y_line)
            return line, ax2, ax3

        ani = animation.FuncAnimation(fig, update, frames=self.n_iters, interval=200, repeat=False)       
        plt.show()
        return self.weights

    def predict(self, X):
        """
        X(numpy array with shape (n,3)) : Testing data
        weights(numpy array ) : The weights produced in the training phase
        
        Returns the algorithms predictions on the testing data
        
        """
        y_predicted = np.ones(X.shape[0])
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, self.weights) 
            y_predicted[idx] = self.activation_func(linear_output)
        return y_predicted, self.weights    






