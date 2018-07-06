
#ifndef LIBIGL_TUTORIALS_FUNCTION_H
#define LIBIGL_TUTORIALS_FUNCTION_H


#include "igl/readOFF.h"
#include <ANN/ANN.h>
#include <iostream>
#include <cmath>
#include <random>
#include "typedefs.h"
#include "igl/invert_diag.h"
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::Vector3d;
class Function {

public:

    double Edistance(Eigen::MatrixXd V1, Eigen::MatrixXd V2);
    Eigen::MatrixXd transfomationMatrix_x(Eigen::MatrixXd V, double theta,double movex,double movey,double z);
    Eigen::MatrixXd transfomationMatrix_y(Eigen::MatrixXd V, double theta,double movex,double movey,double z);
    Eigen::MatrixXd transfomationMatrix_z(Eigen::MatrixXd V, double theta,double movex,double movey,double z);
    Eigen::MatrixXd ICP(Eigen::MatrixXd VA,Eigen::MatrixXd VB);
    Eigen::MatrixXd ICP_plane(Eigen::MatrixXd VA,Eigen::MatrixXd VB,Eigen::MatrixXd VAnormals);
    Eigen::VectorXd meanCurvature(Eigen::MatrixXd V1,Eigen::MatrixXi F1,Eigen::MatrixXd Vnormal,acq::NeighboursT Neighbours);
    acq::NeighboursT computeNeighbours(Eigen::MatrixXi &F1, Eigen::MatrixXd &V1);
    Eigen::VectorXd gaussianCurvature(Eigen::MatrixXd V1,Eigen::MatrixXi F1);
    Eigen::VectorXd NonuniformCurvature(Eigen::MatrixXd V1,Eigen::MatrixXi F1,Eigen::MatrixXd Vnormal);
    Eigen::MatrixXd ReConstruction(Eigen::MatrixXd V1,Eigen::MatrixXi F1,int k);
    Eigen::MatrixXd exSmoothing(Eigen::MatrixXd V1,Eigen::MatrixXi F1,acq::NeighboursT Neighbours,
                                          double lambda,int type,int iteration);
    Eigen::MatrixXd imSmoothing(Eigen::MatrixXd V1,Eigen::MatrixXi F1,
                                double lambda,int iteration);
};


#endif //LIBIGL_TUTORIALS_FUNCTION_H
