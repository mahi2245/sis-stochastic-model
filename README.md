step 1: get training samples
	python3 runX.py [number of training samples]

step 2: train model using training samples
	python3 train.py

step 3: predict using trained model
	python3 predict.py [generations] [sample size] [infected_t1] [infected_t2]




additional step: generate a model comparing stochastic model and ODE model
	python3 model.py --S 90 --I 10 --beta 0.5 --gamma 0.2 --generations 200