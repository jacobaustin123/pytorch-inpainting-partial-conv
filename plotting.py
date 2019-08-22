from visdom import Visdom
import numpy as np

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', port=8050):
        try:
            self.viz = Visdom(port=port)
        except (ConnectionError, ConnectionRefusedError) as e:
            raise ConnectionError("Visdom Server not running, please launch it with `visdom` in the terminal")
    
        self.env = env_name
        self.plots = {}
    
    def clear(self):
        self.plots = {}
        
    def imshow(self, var_name, images):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images)
        else:
            self.viz.images(images, win=self.plots[var_name], env=self.env)

    def plot(self, var_name, split_name, title_name, x, y, xlabel='epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
