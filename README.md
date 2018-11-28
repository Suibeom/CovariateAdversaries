# CovariateAdversaries

Neural networks can figure out what neural networks are thinking. So we'll use them to see if they're thinking about some covariate they shouldn't be thinking about, and use adverarial techniques to confound the gradient in those directions and interfere with the network's ability to learn these unwanted features.

The theoretical idea couldn't be simpler, and it seems to work just fine.

We can use this to force insensitivity of neural networks to cheap but bad features (proxies for race and gender in resumes, obvious foreground/background contrast gradients that work well for white faces and poorly for brown ones, and a host of others.)

We could even use this to establish the importance of certain known features in decision making, similar to the way blanking input coarsely is currently used.
