package org.deeplearning4j;

import org.jfree.chart.*;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.chart.plot.PlotOrientation;

import javax.swing.JFrame;
import java.awt.BorderLayout;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * グラフ作成
 * Created by b1012059 on 2015/05/31.
 */
public class CreateGraph extends JFrame{
    private String graphTitle;
    private String categoryAxisLabel;
    private String valueAxisLabel;

    public CreateGraph(String TITLE, String CATEGORY, String VALUE) {
        this.graphTitle = TITLE;
        this.categoryAxisLabel = CATEGORY;
        this.valueAxisLabel = VALUE;
    }


    public ChartPanel createGraphPanel(int dataCount, ArrayList<Double> tmpError){

        File file = new File("./graphTest.jpeg");
        String str;
        ChartFactory.setChartTheme(StandardChartTheme.createLegacyTheme());
        DefaultCategoryDataset data = new DefaultCategoryDataset();

        for(int i = 0; i < dataCount; i += 20){
            str = Integer.toString(i);
            data.addValue(tmpError.get(i), "二乗誤差", str);
        }





        JFreeChart chart = ChartFactory.createLineChart(
                graphTitle,
                categoryAxisLabel,
                valueAxisLabel,
                data,
                PlotOrientation.VERTICAL,
                true,
                false,
                false);

        try {
            ChartUtilities.saveChartAsJPEG(file, chart, 800, 500);
        } catch (IOException e) {
            e.printStackTrace();
        }

        ChartPanel cpanel = new ChartPanel(chart);
        return cpanel;

        //getContentPane().add(cpanel, BorderLayout.CENTER);
    }

    public ChartPanel createGraphPanel2(int dataCount, ArrayList<Double> tmpError, ArrayList<Double> tmpZero, ArrayList<Double> tmpOne,
                                       ArrayList<Double> tmpTwo, ArrayList<Double> tmpThree, ArrayList<Double> tmpFour,
                                       ArrayList<Double> tmpFive, ArrayList<Double> tmpSix, ArrayList<Double> tmpSeven,
                                       ArrayList<Double> tmpEight, ArrayList<Double> tmpNine){

        File file = new File("./graphTest.jpeg");
        String str;
        ChartFactory.setChartTheme(StandardChartTheme.createLegacyTheme());
        DefaultCategoryDataset data = new DefaultCategoryDataset();

        for(int i = 0; i < dataCount; i += 20){
            str = Integer.toString(i);
            data.addValue(tmpError.get(i), "二乗誤差", str);

            /*if(i == 0){
                data.addValue(tmpZero.get(i), "0", str);
                data.addValue(tmpOne.get(i), "1", str);
                data.addValue(tmpTwo.get(i), "2", str);
                data.addValue(tmpThree.get(i), "3", str);
                data.addValue(tmpFour.get(i), "4", str);
                data.addValue(tmpFive.get(i), "5", str);
                data.addValue(tmpSix.get(i), "6", str);
                data.addValue(tmpSeven.get(i), "7", str);
                data.addValue(tmpEight.get(i), "8", str);
                data.addValue(tmpNine.get(i), "9", str);
            } else if(i == dataCount-1){
                data.addValue(tmpZero.get(tmpZero.size()-1), "0", str);
                data.addValue(tmpOne.get(tmpOne.size()-1), "1", str);
                data.addValue(tmpTwo.get(tmpTwo.size()-1), "2", str);
                data.addValue(tmpThree.get(tmpThree.size()-1), "3", str);
                data.addValue(tmpFour.get(tmpFour.size()-1), "4", str);
                data.addValue(tmpFive.get(tmpFive.size()-1), "5", str);
                data.addValue(tmpSix.get(tmpSix.size()-1), "6", str);
                data.addValue(tmpSeven.get(tmpSeven.size()-1), "7", str);
                data.addValue(tmpEight.get(tmpEight.size()-1), "8", str);
                data.addValue(tmpNine.get(tmpNine.size()-1), "9", str);
            }*/
        }





        JFreeChart chart = ChartFactory.createLineChart(
                graphTitle,
                categoryAxisLabel,
                valueAxisLabel,
                data,
                PlotOrientation.VERTICAL,
                true,
                false,
                false);

        try {
            ChartUtilities.saveChartAsJPEG(file, chart, 800, 500);
        } catch (IOException e) {
            e.printStackTrace();
        }

        ChartPanel cpanel = new ChartPanel(chart);
        return cpanel;

        //getContentPane().add(cpanel, BorderLayout.CENTER);
    }
}
