package org.deeplearning4j;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class OtherExample {
    public static void main(String... args) throws Exception {
        int numFeatures = 614;

        INDArray input = Nd4j.create(2, numFeatures); // have to be at least two or else output layer gradient is a scalar and cause exception
        INDArray labels = Nd4j.create(2, 2);

        INDArray row0 = Nd4j.create(1, numFeatures);
        row0.assign(0.1);

        input.putRow(0, row0);
        labels.put(0, 1, 1); // set the 4th column

        INDArray row1 = Nd4j.create(1, numFeatures);
        row1.assign(0.2);
        input.putRow(1, row1);
        labels.put(1, 0, 1); // set the 2nd column

        DataSet trainingSet = new DataSet(input, labels);

        MersenneTwister gen = new MersenneTwister(123);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(3)
                .weightInit(WeightInit.SIZE)
                .activationFunction("tanh").layerFactory(LayerFactories.getFactory(RBM.class))
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .learningRate(1e-1f)
                .nIn(trainingSet.numInputs()).nOut(trainingSet.numOutcomes()).list(2)
                .hiddenLayerSizes(new int[]{400})
                .override(new ClassifierOverride(3)).build();

        MultiLayerNetwork nn = new MultiLayerNetwork(conf);
        nn.fit(trainingSet);
        INDArray predict2 = nn.output(trainingSet.getFeatureMatrix());
        for (int i = 0; i < predict2.rows(); i++) {
            String actual = trainingSet.getLabels().getRow(i).toString().trim();
            String predicted = predict2.getRow(i).toString().trim();
            System.out.println("actual "+actual+" vs predicted " + predicted);
        }
//        for (int row = 0; row < nn.getOutputLayer().getW().rows(); row++) {
//            System.out.println(nn.getOutputLayer().getW().getRow(row).toString().trim());
//        }
        Evaluation eval = new Evaluation();
        eval.eval(trainingSet.getLabels(), predict2);
        System.out.println("F1: " + eval.f1());
    }
}