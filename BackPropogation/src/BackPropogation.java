//Ahmed Essawi 04/15/2023 Recognize Digits Algorithm
import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class BackPropogation{
    //defining necessary values
    int iterations = 10000;
    double learningRate = 0.5;
    //matrix with rows as first layer of nodes and colums as second layers with data as the weights
    double[][] inputHidden = new double[15][12];
    double[][] hiddenOutput = new double[12][10];
    double[][] trainingData = new double[27][25];
    double[][] validationData = new double[26][25];
    //training and validation data for each digit
    /* 
    double[][] trainingData = {
        {1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0}, //0
        {0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0}, //1
        {0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0}, //1
        {0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0}, //1 
        {1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0}, //2
        {0,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0}, //2
        {1,1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0}, //2
        {1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0}, //3
        {1,1,1,0,0,1,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0}, //3
        {0,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0}, //3
        {1,1,1,0,0,1,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0}, //3
        {1,0,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0}, //4
        {0,0,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0}, //4
        {1,0,0,1,0,1,1,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0}, //4
        {1,0,1,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0}, //4
        {1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0}, //5
        {1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0}, //5
        {1,1,1,1,0,0,1,1,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0}, //5
        {1,1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0}, //6
        {1,1,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0}, //6
        {1,0,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0}, //6
        {1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0}, //7
        {1,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0}, //7
        {1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,1,0}, //8
        {1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1}, //9
        {1,1,1,1,0,1,1,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1}, //9
        {1,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1}  //9
    };
    double[][] validationData = {
        {1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0}, //0
        {0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0}, //1
        {0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0}, //1
        {0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0}, //1
        {1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0}, //2
        {1,1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0}, //2
        {0,1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0}, //2
        {1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0}, //3
        {0,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0}, //3
        {0,1,1,0,0,1,0,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0}, //3
        {1,0,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0}, //4
        {0,0,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0}, //4
        {0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0}, //4
        {1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0}, //5
        {1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0}, //5
        {1,1,0,1,0,0,1,1,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0}, //5
        {1,1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0}, //6
        {1,1,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0}, //6
        {1,0,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0}, //6
        {1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0}, //7
        {1,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0}, //7
        {0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0}, //7
        {1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,1,0}, //8
        {1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1}, //9
        {1,1,1,1,0,1,1,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1}, //9
        {1,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1}  //9
    };
    */
    public static void main(String[] args) throws IOException{
        BackPropogation obj = new BackPropogation();
        //connecting to file
        String fileName = "testData.txt";
        //creating file reader obj
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        //moving reader to the first line
		String line = br.readLine();
		
		int count = 0;
		//loop until there is no longer values in the line
		while(line != null)
		{
            //split the line at the comma
		    String[] values = line.split(",");
		  
            //loop through the values
		    for(int j = 0; j<values.length;j++)
		    {
                //add each value to training data
			     obj.trainingData[count][j] = Double.parseDouble(values[j]);
		    }
            //increase count and go to next line
		    count++;
		    line = br.readLine();
		}
		
        //end the current file reading
		br.close();
		
		fileName = "validationData.txt";
		
        //create new obj for new file
		br = new BufferedReader(new FileReader(fileName));
		
        //move to first line
		line = br.readLine();
		
		count = 0;
		//loop until there is no longer values in the line
		while(line != null)
		{
            //split the line at the comma
		    String[] values = line.split(",");
		    //loop through the values
		    for(int j = 0; j<values.length;j++)
		    {
                //add each value to validation data
			    obj.validationData[count][j] = Double.parseDouble(values[j]);
		    }
            //add to count and move to next line
		    count++;
		    line = br.readLine();
		}
       

        //call all the necessary methods throughout
        BackPropogation.createRandomWeights(obj);
        BackPropogation.training(obj);
        BackPropogation.validation(obj);
        
    }
    //method to generate random weights between -0.2 and 0.2 to start training
    public static void createRandomWeights(BackPropogation obj){
        Random rand = new Random();
        //loop through all of input hidden weights
        for(int i = 0; i<obj.inputHidden.length;i++){
            for(int j = 0; j<obj.inputHidden[0].length; j++){
                //set it equal to a number between -0.2 and 0.2
                obj.inputHidden[i][j] = (rand.nextDouble()/5) - (rand.nextDouble()/5);
            }
        }
        //loop through all of hidden output weights
        for(int i = 0; i<obj.hiddenOutput.length; i++){
            for(int j = 0; j<obj.hiddenOutput[0].length; j++){
                //set it equal to a number between -0.2 and 0.2
                obj.hiddenOutput[i][j] = (rand.nextDouble()/5) - (rand.nextDouble()/5);
            }
        }

    }

    //first half of forward propogation from input to hidden layer
    public static double[] forwardPropogation(double[] inputValues, BackPropogation obj){
        double[] hiddenValues = new double[12];

        //loop through the weights
        for(int i = 0; i<obj.inputHidden[0].length; i++){
            //temp value to sum all weights and inputs
            double value = 0;
            for(int j = 0; j<obj.inputHidden.length; j++){
                //add the wieght for that node multiplied by input
                value += inputValues[j] * obj.inputHidden[j][i];
            }
            //add to hidden values
            hiddenValues[i] = BackPropogation.sigmoidFunction(value);
        }

        return hiddenValues;
    }

    //second half of forward propogation from hidden to output layer
    public static double[] forwardProp2(double[] hiddenValues, BackPropogation obj){
        double[] outputValues = new double[10];
        //loop through weights
        for(int i = 0; i<obj.hiddenOutput[0].length; i++){
            //temp value to sum up
            double value = 0;
            for(int j = 0 ; j<obj.hiddenOutput.length; j++){
                //add the weight and input to the value
                value += hiddenValues[j] * obj.hiddenOutput[j][i];
            }
            //calculate sigmoid function and add to outputValues
            value = BackPropogation.sigmoidFunction(value);
            outputValues[i] = value;
        }

        return outputValues;
    }
    public static double sigmoidFunction(double gX){
        //negate GX
        gX = gX * -1;

        //do e to the power of new gx
        double result = Math.pow(Math.E, gX);

        //plug into the equation
        result = 1 + result;
        result = 1/result;

        return result;
    }

    //calculate error from output to hidden
    public static double outputError(double output, double expected){
        //plug into formula and return error
        double difference = expected - output;
        double error = 1 - output;
        error = output* error;
        error = error * difference;

        return error;
    }
    //calculate error from hidden to input
    public static double hiddenError(double ak, double hiddenvalue){
        //plug into formula and return error
        double error = hiddenvalue * (1 - hiddenvalue) * ak;
        return error;
    }

    //calculate the change needed in the weight
    public static double changeWeight(BackPropogation obj, double error, double input){
        double necessaryChange; 

        //plug into weight formula and return value
        necessaryChange = obj.learningRate * error;
        necessaryChange = necessaryChange * input;
        

        return necessaryChange; 
    }

    //method that truly calculates error and adjusts the weight values
    public static void weightAdjustment(double[] outputValues, double[] expectedOutput, BackPropogation obj, double[] hiddenValues, double[] inputValues){
        double[] outputErrors = new double[10];

        //loop through the length of the output values
        for(int i = 0; i<outputValues.length; i++){
            //createa array of the errors calculated
            outputErrors[i] = outputError(outputValues[i], expectedOutput[i]);
        }

        //loop through all the weights
        for(int i = 0; i<outputErrors.length; i++){
            for(int j = 0; j<hiddenValues.length; j++){
                //change the weight using the necesary value
                obj.hiddenOutput[j][i] += changeWeight(obj, outputErrors[i], hiddenValues[j]);
            }
        }


        double[] ak = new double[12];

        //loop through ak to calcualte all alpha values
        for(int i = 0; i<ak.length; i++){
            double temp = 0;
            for(int j = 0; j<outputErrors.length; j++){
                //add to temp the forumla for aplha
                temp += obj.inputHidden[i][j] * outputErrors[j];
            }
            //set temp to alpha value
            ak[i] = temp;
        }

        double[] hiddenErrors = new double[12];

        //loop through all hidden and calcaulte the error for each node
        for(int i = 0; i<hiddenErrors.length; i++){
            hiddenErrors[i] = hiddenError(ak[i], hiddenValues[i]);
        }

        //loop through all the hidden errors and change the weights
        for(int i = 0; i<hiddenErrors.length; i++){
            for(int j =0; j<inputValues.length; j++){
                obj.inputHidden[j][i] += changeWeight(obj, hiddenErrors[i], inputValues[j]);
            }
        }
    }
    public static void training(BackPropogation obj){
        Random rand = new Random();
        //loop through amount of iterations
        for(int i = 0; i<obj.iterations; i++){
            //random index between 0 and 25
            int index = rand.nextInt(26);
            double[] inputValues = new double[15];
            double[] expectedOutput = new double[10];

            //take first 15 values as part of input
            for(int j = 0; j<15; j++){
                inputValues[j] = obj.trainingData[index][j];
            }
            int count = 0;
            //take last 10 values as part of output
            for(int j = 15; j<obj.trainingData[0].length; j++){
                expectedOutput[count] = obj.trainingData[index][j];
                count++;
            }

            //send values to forward propogation
            double[] hiddenValues = BackPropogation.forwardPropogation(inputValues, obj);
            double[] outputValues = BackPropogation.forwardProp2(hiddenValues, obj);

            //call weight adjustment method
            BackPropogation.weightAdjustment(outputValues, expectedOutput, obj, hiddenValues, inputValues);
        }
    }

    //method to validate if weights are working correctly
    public static void validation(BackPropogation obj){
        //loop through amount needed to be tested
        for(int i = 0; i<10; i++){
            Random rand = new Random();
            //random index from 0 to 25
            int index = rand.nextInt(26);
            double[] inputValues = new double[15];
            double[] expectedOutput = new double[10];

            //create input values
            for(int j = 0; j<15; j++){
                inputValues[j] = obj.validationData[index][j];
            }
            int count = 0;
            //created expected output values
            for(int j = 15; j<obj.validationData[0].length; j++){
                expectedOutput[count] = obj.validationData[index][j];
                count++;
            }

            //call the forward propogation
            double[] hiddenValues = BackPropogation.forwardPropogation(inputValues, obj);
            double[] outputValues = BackPropogation.forwardProp2(hiddenValues, obj);

            //use threshhold to move output values to either 0 or 1
            for(int j = 0; j<outputValues.length; j++){
                if(outputValues[j] >= 0.5){
                    outputValues[j] = 1;
                }
                else{
                    outputValues[j]= 0;
                }
            }

            //print output and expected output for each
            System.out.println(Arrays.toString(outputValues));
            System.out.println(Arrays.toString(expectedOutput));
            System.out.println("");
        }
    }
}