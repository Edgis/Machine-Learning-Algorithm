/**
 * Random-Forest implementation with JAVA 
 * @Author EDGIS
 * @Contact guoxianwhu@foxmail.com
 */

package com.rf.real;

import java.util.ArrayList;

public class MainRun {
    @SuppressWarnings("static-access")
    public static void main(String args[]){

        String trainPath = "C:\\Users\\kwok\\Desktop\\rf_data\\Data.txt";
        String testPath = "C:\\Users\\kwok\\Desktop\\rf_data\\Test.txt";
        int numTrees = 100;

        DescribeTrees DT = new DescribeTrees(trainPath);
        ArrayList<int[]> Train = DT.CreateInput(trainPath);

        DescribeTrees DT2 = new DescribeTrees(testPath);
        ArrayList<int[]> Test = DT2.CreateInput(testPath);
        int categ = 0;

        //the num of labels
        int trainLength = Train.get(0).length;
        for(int k=0; k<Train.size(); k++){
            if(Train.get(k)[trainLength-1] < categ)
                continue;
            else{
                categ = Train.get(k)[trainLength-1];
            }
        }

        RandomForest Rf = new RandomForest(numTrees, Train, Test);
        Rf.C = categ;//the num of labels
        Rf.M = Train.get(0).length-1;//the num of Attr
        //属性扰动，每次从M个属性中随机选取Ms个属性，Ms = ln(m)/ln(2)
        Rf.Ms = (int)Math.round(Math.log(Rf.M)/Math.log(2) + 1);//随机选择的属性数量
        Rf.Start();
    }
}
