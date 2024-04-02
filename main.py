import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import points
import matplotlib.gridspec as gridspec
import perceptron 
import adaline
import mlp_from_lib
import mlp_custom

def recall_perceptron(perceptron_anim,X_test):
    """
    perceptron_anim (PerceptronAnimation object) : Used to predict the classes of X_test data
    X_test (numpy array with shape (n,3)): The data

    Takes a number of points and uses the predict() function to categorize them and then plots 4 subplots. 
    """
    
    predicted,weights = perceptron_anim.predict(X_test)
    fig,gs,ax1,ax2,axN= perceptron.create_subplots()
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1]) 
    #Set titles
    ax1.set_title('Points')
    ax2.set_title('Training')
    ax3.set_title('Output graph')
    ax4.set_title('Targets and predictions')
    
    #First 3 plots and the 2's line
    x_line,y_line = perceptron.draw(X_test,weights,ax1,ax2,ax3,predicted)
    ax2.set_ylim(-0.1,)
    line, = ax2.plot(x_line, y_line, 'r-', lw=2)
    #Plot 4
    #ax4.scatter(X_test[:,0],X_test[:,1] , marker='o', c='blue', label='Stoxoi')
    ax4.scatter(np.arange(0, n), predicted, marker='x', c='black', label='Predictions perceptron')
    ax4.scatter(np.arange(0, n//2), np.zeros(n//2), marker='o', c='red', label='Points')
    ax4.scatter(np.arange(n//2, n), np.ones(n//2), marker='o', c='red')
    ax4.legend()
    

    plt.show()

def recall_adaline(n, adaline_anim,X_test, weights, weights_mse):
    """
    adaline_anim (AdalineAnimation object) : Used to predict the classes of X_test data
    X_test (numpy array with shape (n,3)): The data

    Takes a number of points and uses the predict() function to categorize them and then plots 4 subplots. 
    """
    
    predicted = adaline_anim.predict(X_test, weights)
    fig,ax1,ax2,ax3,ax4 = adaline.create_subplots()
    x_line,y_line = adaline.draw(X_test,weights,ax1,ax2,ax3,predicted)

    ax1.set_title('Targets divided by mse line')
    ax2.set_title('Predictions with adaline line')
    
    #Plot 1- predicted me eutheia mse
    x_line_mse = X_test[:,0]
    y_line_mse = -(weights_mse[0]*x_line_mse - weights_mse[2])/weights_mse[1]
    line_mse = ax1.plot(x_line_mse,y_line_mse, 'r-', lw=2)
    
    #Plot 2 
    line, = ax2.plot(x_line, y_line, 'r-', lw=2)

    #Plot 3
    ax3.set_title('Targets and predictions adaline')
    ax3.cla()
    ax3.scatter(np.arange(0, n), predicted, marker='x', c='black', label='Predictions adaline')
    ax3.scatter(np.arange(0, n//2), np.zeros(n//2), marker='o', c='red', label='Points')
    ax3.scatter(np.arange(n//2, n), np.ones(n//2), marker='o', c='red')
    ax3.legend()

    #Plot 4
    ax4.set_title('Targets and predictions mse ')
    predicted_mse = adaline_anim.predict(X_test, weights_mse)
    ax4.scatter(np.arange(0, n), predicted_mse, marker='x', c='black', label='Predictions mse')
    ax4.scatter(np.arange(0, n//2), np.zeros(n//2), marker='o', c='red', label='Points')
    ax4.scatter(np.arange(n//2, n), np.ones(n//2), marker='o', c='red')
    ax4.legend()

    plt.show()

def recall_mlp_lib(mlp_anim,X_test, mlp):
    
    n = X_test.shape[0]
    predicted = mlp_anim.predict(X_test, mlp)
    fig,ax1,ax2,ax3,ax4= mlp_from_lib.create_subplots()
    
    #Set titles
    ax1.set_title('Points')
    ax2.set_title('Training')
    ax3.set_title('Targets and predictions mlp')
    ax3.legend()
    ax4.set_title('Targets and predictions')
    
    #Draws ax1,ax2,ax3
    mlp_from_lib.draw(X_test,ax1,ax2,ax3,predicted)
      

    #Plot 4
    ax4.scatter(np.arange(0, n), predicted, marker='x', c='black', label='Predictions mlp')
    ax4.scatter(np.arange(0, n//2), np.zeros(n//2), marker='o', c='red', label='Prwtupa')
    ax4.scatter(np.arange(n//2, n), np.ones(n//2), marker='o', c='red')
    ax4.legend()

    plt.show()

def recall_mlp_custom(mlp_anim,X_test):
    
    n = X_test.shape[0]
    predicted = mlp_anim.predict(X_test)
    fig,ax1,ax2,ax3,ax4= mlp_from_lib.create_subplots()
    
    #Set titles
    ax1.set_title('Points')
    ax2.set_title('Training')
    ax3.set_title('Targets and predictions mlp')
    ax3.legend()
    ax4.set_title('Targets and predictions')
    
    #Draws ax1,ax2,ax3
    mlp_from_lib.draw(X_test,ax1,ax2,ax3,predicted)
       
    #Plot 4
    ax4.scatter(np.arange(0, n), predicted, marker='x', c='black', label='Predictions mlp')
    ax4.scatter(np.arange(0, n//2), np.zeros(n//2), marker='o', c='red', label='Points')
    ax4.scatter(np.arange(n//2, n), np.ones(n//2), marker='o', c='red')
    ax4.legend()

    plt.show()

if __name__ == "__main__":
    flag = 1
    while flag==1:
        print("\n1: Perceptron")
        print("2: Adaline")
        print("3: MLP")
        algorithm_choice = int(input("Choose algorithm: "))
        #Choose number of prwtupa
        n = int(input("Give value multiple of 8: "))
        print("\n1: Linearly Separable")
        print("2: Non-linearly separable (corner)")
        print("3: Non-linearly separable (centre)")
        print("4: Non-linearly separable (class 1 inside class 2)")
        print("5: XOR")
        print("6: End")
        choice = int(input("Choose points: "))
        if choice == 1:
            X_train = points.linearly_separable(n)
            X_test = points.linearly_separable(n)
        elif choice == 2:
            X_train = points.non_linearly_separable_corner(n)
            X_test = points.non_linearly_separable_corner(n)
        elif choice == 3:
            X_train = points.non_linearly_separable_centre(n)
            X_test = points.non_linearly_separable_centre(n)
        elif choice == 4:
            X_train = points.non_linearly_separableE(n)
            X_test = points.non_linearly_separableE(n)
        elif choice == 5:
            X_train = points.XOR(n)
            X_test = points.XOR(n)
        elif choice == 6:
            flag = 0
            break
        epochs = int(input("Number of epochs: "))
        learning_rate = float(input("Learning rate: "))
        
        
        if algorithm_choice == 1:
            y_train = np.zeros((n, 1))
            y_train[n//2:n, ] = np.ones((n//2, 1))
            perceptron_anim = perceptron.PerceptronAnimation(n_iters=epochs, lr=learning_rate )
            perceptron_anim.fit(X_train, y_train)
            recall_perceptron(perceptron_anim,X_test)
        elif algorithm_choice == 2:
            min_error = float(input("Min error: "))
            y_train = np.zeros((n, 1))
            y_train[n//2:n, ] = np.ones((n//2, 1))
            y_train[:n//2, ] = -1
            adaline_anim = adaline.AdalineAnimation(n_iters=epochs, lr=learning_rate,min_error_for_termination = min_error)
            weights, weights_mse = adaline_anim.fit(X_train, y_train)
            recall_adaline(n, adaline_anim, X_test, weights, weights_mse)
        elif algorithm_choice == 3:
            X_train = X_train[:, :2]
            X_test = X_test[:, :2]
            min_error = float(input("Min error: "))
            hidden_neurons = int(input("Give number of hidden neurons : "))
            print("\n1: Gradient Descent")
            print("2: Stochastic Gradient Descent")
            print("3: Adam")
            print("4: Telos")
            method = int(input('Choose method: '))
            y_train = np.zeros((n, ))
            y_train[n//2:n, ] = np.ones((n//2, ))
            y_train[:n//2, ] = 0
            if method == 1:
                print("1: relu")
                print("2: tahn")
                print("3: logistic")
                hidden_activation = int(input('Choose hidden layer activation method: '))
                print("1: relu")
                print("2: tahn")
                print("3: logistic")
                output_activation = int(input('Choose output layer activation method: '))
                mlp = mlp_custom.MLP_custom(input_size=2, hidden_size=hidden_neurons, output_size=2, hidden_activation_choice=hidden_activation, output_activation_choice=output_activation, min_error_for_termination=min_error)
                mlp.train_and_animate(X_train, y_train, epochs)
                recall_mlp_custom(mlp, X_test)
            elif method == 2:
                mlp_anim = mlp_from_lib.MLPAnimation('sgd', n_iters=epochs, lr=learning_rate,min_error_for_termination = min_error, hidden_layer_size=hidden_neurons)
                weights, mlp = mlp_anim.fit(X_train, y_train)
                recall_mlp_lib(mlp_anim, X_test, mlp)
            elif method == 3:
                mlp_anim = mlp_from_lib.MLPAnimation('adam', n_iters=epochs, lr=learning_rate,min_error_for_termination = min_error,  hidden_layer_size=hidden_neurons)
                weights, mlp = mlp_anim.fit(X_train, y_train)
                recall_mlp_lib(mlp_anim, X_test, mlp)
            elif method == 4:
                flag = 0
                break
             

            
            