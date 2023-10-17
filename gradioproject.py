import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

def linearregression(a_gradin,ite_gradin):
    # generate random data-set
    #np.random.seed(0) # choose random seed (optional)
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.rand(100, 1)

    J = 0 # initialize J, this can be deleted once J is defined in the loop
    w = np.matrix([np.random.rand(),np.random.rand()]) # slope and y-intercept
    # a = 0.1 # learning rate step size
    a = a_gradin;  #user input from gradio
    # ite = 10 # number of training iterations
    ite = ite_gradin;  #user input from gradio

    Losslist = []      # list to store loss values
    dd = [] 

    ## Write Linear Regression Code to Solve for w (slope and y-intercept) Here ##
    for p in range (ite):
        for i in range(len(x)):
            # Calculate w and J here
            x_vec = np.matrix([x[i][0],1]) # Setting up a vector for x (x_vec[j] corresponds to w[j])
            h = w * x_vec.T # h = (define h here) ## Hint: you may need to transpose x or w by adding .T to the end of the variable
            w = w - a *(h - y [i])*x_vec # w = (define w update iteration here)
            J = 0.5*(h-y[i])**2 # J = (loss equation here)
        
        J = J.item()
    
        Losslist.append(J)    # puts the loss values into the created list 
        dd.append(p)
        
        print('Loss:', J)

    ## if done correctly the line should be in line with the data points ##

    print('f = ', w[0,0],'x + ', w[0,1])

    # plot
    # Create a loss plot
    loss_fig = plt.figure()
    plt.plot(dd, Losslist, linestyle='solid')
    plt.xlabel('x')
    plt.ylabel('y')

    # Create a data plot
    data_fig = plt.figure()
    plt.scatter(x,y,s=10)
    plt.plot(x, w[0,1] + (w[0,0] * x), linestyle='solid')
    plt.xlabel('x')
    plt.ylabel('y')
    
    return J, data_fig, loss_fig

    

with gr.Blocks() as demo:
    gr.Markdown("Linear Regression")
    with gr.Row():
        input = [gr.Slider(label="Learning Rate",value=0.1,maximum=0.9,minimum=0.1,step=0.10),gr.Slider(label="Number of Training Interations",value=0,maximum=100,minimum=0,step=1)]
    with gr.Row():
        out = gr.Textbox(label="Loss")
    with gr.Row():
        data_plot = gr.Plot()
        loss_plot = gr.Plot()
    with gr.Row():
        btn = gr.Button("Run")
    btn.click(fn=linearregression, inputs=input, outputs=[out,data_plot, loss_plot])
    

demo.launch()