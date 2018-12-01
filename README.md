# CovariateAdversaries

Neural networks can figure out what neural networks are thinking. So we'll use them to see if they're thinking about some covariate they shouldn't be thinking about, and use adverarial techniques to confound the gradient in those directions and interfere with the network's ability to learn these unwanted features.

We can use this to force insensitivity of neural networks to cheap but bad features (proxies for race and gender in resumes, obvious foreground/background contrast gradients that work well for white faces and poorly for brown ones, and a host of others.)

We could even use this to establish the importance of certain known features in decision making, similar to the way blanking input coarsely is currently used.

Pitting the neural networks DIRECTLY against each other is the obvious first solution, but in my experiments so far it seems like one network will just kill the other and it'll need to be reinitialized. Some kind of intelligent moderator will help with this.
