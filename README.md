# Machine Learning Surrogates for Finite Element Analysis
## Overview
This project develops machine-learning-based surrogate models for finite element analysis and evaluates their performance for various soft grippers. The workflow combines automated CAD generation for the tested grippers, finite element preprocessing, simulation data generation, and the development and comparison of multiple machine learning models across three test cases.

## Project Workflow and Methodology
### 1. Parametric CAD generation using automated scripts:
Using FreeCAD's native spreadsheet feature, a script was used to automate the generation of CAD models, changing parameters for each specific gripper. For instance, with the Finray gripper, the variables were baseSize, externalBeamWidth, internalBeamSpacing, internalBeamSize, whereas for the A-shaped gripper it was beamLength, beamHeight, externalBeamWidth, internalBeamWidth. Two methods were used to generate combinations of these variables. The first was to vary each of these parameters across a defined range, using a step size appropriate to its magnitude. For example, internal beam spacing on the Fin-ray gripper was varied from 30mm to 100mm with a step size of 10mm. This yielded around four thousand simulations for each gripper. The second was to use the Latin Hypercube Sampling (LHS) method to select combinations so as to fill as much of the sample space as possible. This yielded around three hundred simulations for each gripper. The accuracy of models trained on each dataset was compared. A secondary dataset was also generated with more extreme values to test extrapolation abilities.
### 2. Meshing, gripper-specific boundary-condition assignment and finite element simulation:
Meshing was done automatically using the gmsh library. To automate boundary conditions, I used mathematical methods to look for specific features of each gripper, calculate the location of the faces of the feature, and constrain them using Python. For instance, with the VSG developed by my mentor certain cylindrical surfaces were searched for to set boundary conditions, whereas for Finray grippers faces with a certain direction vector within a certain 3D region were examined to apply the load. This approach limited transferability, which could be a focus in further works, but was effective for the purpose of this investigation.
### 3. Finite element simulation:
The CalculiX CrunchiX (ccx) was used in a script to automatically do finite element analysis, after which the max displacement was extracted and recorded alongside the parameters for that specific simulation.
### 4. Training and evaluation of multiple surrogate models:
Three main models were chosen and compared to see which was the quickest and most accurate: Random Forest, XGBoost and neural networks. Linear regression was also tested as a control. 

Random Forest was chosen for its being straightforward to train and ability to handle outliers or messy data. Random Forest also easily shows which features were most key for the final prediction. Random Forest is also less likely to overfit, however is poor at extrapolating beyond the data range given, which is important with such applications, as they often focus on predictions for robots that have not yet been built. 

XGBoost was chosen because of its sequential use of decision trees, which focus on correcting the errors of previous trees. The format of the data (a csv file containing inputs and outputs) was suitable as XGBoost has a high accuracy on such forms of data. However, XGBoost is easy to overfit, which was predicted to limit its accuracy.

Neural networks are a useful model for complex, nonlinear relationships, however, they require more data to perform well. They can also be more difficult to interpret than tree-based models. It was aimed to discover whether the improvement in prediction accuracy for neural networks was sufficient to justify its increased complexity and large data needs.

To train and evaluate each model, the datasets were split into training and testing sets; the testing sets comprised 20% of the full dataset. The models were trained using progressively larger subsets of the training data, e.g. 25, 50, 100, 150 etc. cases. After each stage of training, the model was given a set of test cases to see how its performance had changed.
### 5. Comparison of model performance across test cases:
The models were run on the datasets and the results were compared for accuracy based on training size. For this, the statistical metrics of MAE, RMSE and R<sup>2</sup> were collected.
MAE: The Mean Absolute Error demonstrated how far off the model was on average
RMSE: The Root Mean Squared Error highlighted the effect of extreme outliers
R<sup>2</sup>: The Coefficient of Determination demonstrated how well the predictions matched the physical behaviour patterns from a high-level perspective 

## Results
The results can be found in the results folder. Specific conclusions include:
A smaller LHS dataset performed much better than a larger structured sampling dataset:
Neural network overall predicted behavior the most accurately: this reflects their ability to predict continuous nonlinear relationships 
Neural network was also optimal for extrapolation usages: this is in line with the limitations of decision-based models, which do not predict values outside of the training set

## Repository Structure

CAD Automation: CAD parametric script
Meshing: Automatic meshing scripts, alongside automatic boundary condition applicator
Run: Runs the Calculix FEA and generates results for inputted set of samples
Machine Learning Model: Including main.py, which was used to generate results, as well as the different models

## Reproducibility
An example of a sample parametric dataset for a Fin-ray model has been included, as well as example 3D model/input files (STEP, INP, etc.) and results.
