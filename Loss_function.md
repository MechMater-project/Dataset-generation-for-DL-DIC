Aspects to consider for the choice of a loss function:

1. The accuracy of the results
2. The convergence efficiency
   
-  Type of Loss function 
	-  Mean squared error (MSE)
   
	$L_{MSE}=\frac{1}{N}\sum_{i=1}^n (u_i-\hat{u}_i)^2$

	+Smooth and differentiable everywhere. Strongly penalize large errors. Stable gradients during training.
	-Very sensitive outliers. Can oversmooth the solution. A few very bad pixels can dominate the loss (since every pixel have the same weight).

	- Mean absolute error (MAE)
   
	$L_{MAE}=\frac{1}{N}\sum_{i=1}^n |u_i-\hat{u}_i|$

	+Sensitivity of MAE to outliers is lower than MSE, often preserves discontinuities better.
	-Gradient is discontinuous at zero, convergence may be slower, provides weaker gradients for large errors (whatever the error the gradient is the same. So for MAE an error of 0.01 in a region of 1% strain and in a region of 100% strain will contribute equally to the loss function)

	- Average endpoint error (AEE)
   
	$	L_{AEE}=\frac{1}{N}\sum_{i=1}^n \sqrt{(u_i-\hat{u}_i)^2}$

	+Measures vector error rather than component wise error,
	-Same robustness issues as MAE, does not distinguish angular error and amplitude error
	Commonly used in the field of optical flow

	- Root mean square error (RMSE)
   
	$
	L_{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^n (u_i-\hat{u}_i)^2}

	MSE and RMSE similar from a mathematical perspective as their derivatives differ by only a fixed coefficient. So same + and -. 
	RMSE is used for classic DIC, which explains its use for DL-DIC.

	- Relative loss error
	
	$L_{RLE}=\frac{1}{N}\sum_{i=1}^n \frac{|u_i-\hat{u}_i|}{|u|+\xi}$

	Interesting when there is both large and small deformation involved or strong variation of strain for instance with matrix/filler, crack, strain localization
	
	
- If the objective is **displacement accuracy in pixels**, MAE or AEE are perfectly reasonable.
- If the objective is **capturing both low-strain and high-strain regions with similar relative accuracy**, relative strain losses become very attractive.
- If the objective is **strain localization**, a combination of absolute and relative terms is often the most balanced approach.
	
- Why we don't need to add the gradient of $u$ in the loss function?
	The network architecture itself acts as an implicit regularizer. CNNs have strong built-in biases:
	-convolution kernels are local,
	-neighboring pixels share weights,
	-pooling creates large receptive fields,
	-multi-scale pyramids encourage smooth flow.
  As a consequence, CNNs naturally prefer smooth displacement fields.
  Therefore, there is no risk of checkerboard as in for instance shape optimization.
  


