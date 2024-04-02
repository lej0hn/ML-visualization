from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import points
import matplotlib.gridspec as gridspec


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

class MLP_custom:
    def __init__(self, input_size, hidden_size, output_size, hidden_activation_choice, output_activation_choice, min_error_for_termination):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_activation_choice = hidden_activation_choice
        self.output_activation_choice = output_activation_choice
        self.min_error_for_termination = min_error_for_termination
        
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
    
    def forward(self, X):
        self.hidden_layer = np.dot(X, self.W1) + self.b1
        if self.hidden_activation_choice == 1:
            self.hidden_activation = self.tanh(self.hidden_layer)
        elif self.hidden_activation_choice == 2:
             self.hidden_activation = self.logistic(self.hidden_layer)
        elif self.hidden_activation_choice == 3:
             self.hidden_activation = self.relu(self.hidden_layer)
        self.output_layer = np.dot(self.hidden_activation, self.W2) + self.b2
        if self.output_activation_choice == 1:
            self.output_activation = self.tanh(self.output_layer)
        elif self.output_activation_choice == 2:
             self.output_activation = self.logistic(self.output_layer)
        elif self.output_activation_choice == 3:
             self.output_activation = self.relu(self.output_layer)
        return self.output_activation
    
    def relu(self, X):
        return np.maximum(0, X)
    
    def tanh(self, X):
        return np.tanh(X)
    
    def logistic(self, X):
        return 1 / (1 + np.exp(-X))

    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        # Compute gradients of output layer
        d_output = self.output_activation.copy()
        d_output[range(m), y.astype(int)] -= 1
        d_output /= m
        
        dW2 = np.dot(self.hidden_activation.T, d_output)
        db2 = np.sum(d_output, axis=0)
        
        # Compute gradients of hidden layer
        try:
            d_hidden = np.dot(d_output, self.W2.T) * (1 - np.power(self.hidden_activation, 2))
        except:
             pass
        
        dW1 = np.dot(X.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0)
        
        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def plot_decision_boundary(self, X, y, epoch, ax2, ax3, ax4, mse):
        n = X.shape[0]  

        h = 0.01  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = self.forward(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        predictions = np.argmax(self.forward(X), axis=1)
        
        #Subplot 2
        ax2.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
        ax2.scatter(X[:, 0], X[:, 1], c=predictions, cmap=plt.cm.Spectral)
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_title('Decision Boundary (Epoch {})'.format(epoch))
        
        #Subplot 3
        ax3.cla()
        ax3.set_title('Graphima eksodwn protupwn')
        ax3.scatter(np.arange(0, n//2), predictions[:n//2], marker='x', c='black', label='0')
        ax3.scatter(np.arange(n//2 , n), predictions[n//2:], marker='o', c='magenta', label='1')
        ax3.legend()

        #Plot 4
        ax4.set_title(f'Mean Squared Error : Epoxh - {epoch}')
        ax4.plot(mse, '-', color='k')
    
    def mse(self, y_true, y_pred):
            return np.mean(np.square(y_true - y_pred))
    
    def train_and_animate(self, X, y, num_epochs):
            fig,ax1,ax2,ax3,ax4 = create_subplots()
            ax2.set_title('Training')
        
            ax3.set_title('Graphima eksodwn protupwn')
            ax3.cla()
            mse_values = []

            #Subplot 1
            points.plot_points(X,ax1)
            ax1.set_title('points')

            
            def update(epoch):
                ax2.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
                ax2.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
                self.forward(X)
                self.backward(X, y, learning_rate=0.1)
                mse = self.mse(y, np.argmax(self.forward(X), axis=1))
                if self.min_error_for_termination > mse :
                            ani.event_source.stop()
                mse_values.append(mse)
                ax2.cla()
                self.plot_decision_boundary(X, y, epoch, ax2, ax3,ax4, mse_values)
            
            ani = FuncAnimation(fig, update, frames=num_epochs, interval=50)
            plt.show()

        
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
