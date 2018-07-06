
#include "Function.h"
#include "normalEstimation.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/ArpackSupport>
#include <Eigen/SparseCore>
#include "spectra/GenEigsSolver.h"
#include "spectra/MatOp/SparseGenMatProd.h"
#include <iostream>
#include <igl/eigs.h>
#include "nanogui/formhelper.h"
#include "nanogui/screen.h"
#include "igl/jet.h"
#include "igl/copyleft/cgal/mesh_boolean.h"
#include "igl/copyleft/cgal/intersect_other.h"
#include "igl/unique.h"
#include "igl/triangle_triangle_adjacency.h"
#include <iostream>
#include <cmath>
#include <random>
#include <ANN/ANN.h>

using namespace Eigen;
using namespace Spectra;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using namespace std;
using std::min;

double Function::Edistance(Eigen::MatrixXd V1, Eigen::MatrixXd V2){
    double d1,d2,d3;
    double dis=0;
    for(int i=0; i<V1.rows(); i++){
        d1=V1(i,0)-V2(i,0);
        d2=V1(i,1)-V2(i,1);
        d3=V1(i,2)-V2(i,2);
        dis += sqrt(pow(d1,2)+pow(d2,2)+pow(d3,2));
    }
    return dis;
}
Eigen::MatrixXd Function::transfomationMatrix_x(Eigen::MatrixXd V, double theta,double movex,double movey,double movez){
    Eigen::MatrixXd VT;
    Eigen::Matrix4d H;
    // Set the transformation Matrix
    H<<1,0,0,movex,
            0,cos(theta),sin(theta),movey,
            0,-sin(theta),cos(theta),movez,
            0,0,0,1;
    // Set n*4 matrix of original matrix V
    Eigen::MatrixXd Vtrans(V.rows(), 4);
    Vtrans<<V,Eigen::MatrixXd::Ones(V.rows(),1);
    // Use the transformation matrix to get the result
    Vtrans=(H*Vtrans.adjoint()).adjoint();
    // Adopt the first three cols
    VT=Vtrans.topLeftCorner(Vtrans.rows(),3);
    return VT;
}

Eigen::MatrixXd Function::transfomationMatrix_y(Eigen::MatrixXd V, double theta,double movex,double movey,double movez){
    Eigen::MatrixXd VT;
    Eigen::Matrix4d H;
    // Set the transformation Matrix
    H<<cos (theta),0,sin(theta),movex,
            0,1,0,movey,
            -sin(theta),0,cos(theta),movez,
            0,0,0,1;
    // Set n*4 matrix of original matrix V
    Eigen::MatrixXd Vtrans(V.rows(), 4);
    Vtrans<<V,Eigen::MatrixXd::Ones(V.rows(),1);
    // Use the transformation matrix to get the result
    Vtrans=(H*Vtrans.adjoint()).adjoint();
    // Adopt the first three cols
    VT=Vtrans.topLeftCorner(Vtrans.rows(),3);
    return VT;
}
Eigen::MatrixXd Function::transfomationMatrix_z(Eigen::MatrixXd V, double theta,double movex,double movey,double movez){
    Eigen::MatrixXd VT;
    Eigen::Matrix4d H;
    // Set the transformation Matrix
    H<<cos(theta),sin(theta),0,movex,
            -sin(theta),cos(theta),0,movey,
            0,0,1,movez,
            0,0,0,1;
    // Set n*4 matrix of original matrix V
    Eigen::MatrixXd Vtrans(V.rows(), 4);
    Vtrans<<V,Eigen::MatrixXd::Ones(V.rows(),1);
    // Use the transformation matrix to get the result
    Vtrans=(H*Vtrans.adjoint()).adjoint();
    // Adopt the first three cols
    VT=Vtrans.topLeftCorner(Vtrans.rows(),3);
    return VT;
}

Eigen::MatrixXd Function::ICP(Eigen::MatrixXd VA,Eigen::MatrixXd VB){
    int dim,numA,numB;
    dim=VA.cols();
    numA=VA.rows();
    numB=VB.rows();
    // Set the ANNpoint Array of matrix A and B
    ANNpointArray VA_points;
    ANNpointArray VB_points;
    VA_points=annAllocPts(numA,dim);
    VB_points=annAllocPts(numB,dim);
    // Give the value of VA and VB to ANNpointArray
    ANNpoint point;
    for (int i=0;i<numA;i++){
        point=annAllocPt(dim);
        point[0]=VA(i,0);
        point[1]=VA(i,1);
        point[2]=VA(i,2);
        VA_points[i]=point;
    }
    for (int i=0;i<numB;i++){
        point=annAllocPt(dim);
        point[0]=VB(i,0);
        point[1]=VB(i,1);
        point[2]=VB(i,2);
        VB_points[i]=point;
    }
    // Nearest neighbor index and distance
    ANNidxArray neighbourindex=new ANNidx[1];
    ANNdistArray neighbourdistance=new ANNdist[1];

    Eigen::MatrixXd MatrixA,MatrixB,RoatationMatrix;
    Eigen::Vector3d TranslationMatrix;
    int minnum=std::min(numA,numB);
    MatrixA.setZero(minnum,dim);
    MatrixB.setZero(minnum,dim);

    ANNkd_tree* kdTree;
    kdTree=new ANNkd_tree(VB_points,numB,dim);
    for(int i=0;i<minnum;i=i+1) {
        ANNpoint point = VA_points[i];
        kdTree->annkSearch(
                point,             // the query point
                1,                 // number of near neighbors to return
                neighbourindex,    // nearest neighbor indices (returned)
                neighbourdistance, // the approximate nearest neighbor
                0);                // the error bound
        MatrixA.row(i)=VA.row(i);
        MatrixB.row(i)=VB.row(*neighbourindex);
    }

    Eigen::Vector3d sumA,sumB;
    sumA=MatrixA.colwise().sum();
    sumB=MatrixB.colwise().sum();
    Eigen::MatrixXd Aa,Bb;
    Eigen::Vector3d AveA,AveB;
    AveA=(sumA/numA).transpose();
    AveB=(sumB/numB).transpose();
    Aa=MatrixA-AveA.transpose().replicate(numA,1);
    Bb=MatrixB-AveB.transpose().replicate(numB,1);
    Eigen::MatrixXd Cc;
    Cc = Aa.transpose()*Bb;
    // Use the JacobiSVD
    Eigen::MatrixXd svdU,svdV;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Cc,Eigen::ComputeThinU | Eigen::ComputeThinV);
    svdU=svd.matrixU();
    svdV=svd.matrixV();
    // Calculate the reotation and translation matrix
    RoatationMatrix=svdV*svdU.transpose();
    TranslationMatrix=AveB-RoatationMatrix*AveA;
    Eigen::MatrixXd V;
    V=(RoatationMatrix*VA.transpose()).transpose()+TranslationMatrix.replicate(1,numA).transpose();
    return V;
}

Eigen::MatrixXd Function::ICP_plane(Eigen::MatrixXd VA,Eigen::MatrixXd VB,Eigen::MatrixXd normal){
    int dim,numA,numB;
    dim=VA.cols();
    numA=VA.rows();
    numB=VB.rows();
    ANNpointArray VA_points;
    ANNpointArray VB_points;
    VA_points=annAllocPts(numA,dim);
    VB_points=annAllocPts(numB,dim);
    // Give the value of VA and VB to ANNpointArray
    ANNpoint point;
    for (int i=0;i<numA;i++){
        point=annAllocPt(dim);
        point[0]=VA(i,0);
        point[1]=VA(i,1);
        point[2]=VA(i,2);
        VA_points[i]=point;
    }
    for (int i=0;i<numB;i++){
        point=annAllocPt(dim);
        point[0]=VB(i,0);
        point[1]=VB(i,1);
        point[2]=VB(i,2);
        VB_points[i]=point;
    }
    // Nearest neighbor index and distance
    ANNidxArray neighbourindex=new ANNidx[1];
    ANNdistArray neighbourdistance=new ANNdist[1];
    int minnum=std::min(numA,numB);

    Eigen::MatrixXd Normals;
    Normals.setZero(minnum,3);
    Eigen::MatrixXd MatrixA,MatrixB,Cc,svdU,svdV;
    MatrixA.setZero(minnum,dim);
    MatrixB.setZero(minnum,dim);

    ANNkd_tree* kdTree;
    kdTree=new ANNkd_tree(VA_points,numA,dim);
    for(int i=0;i<minnum;i=i+1) {
        ANNpoint point = VB_points[i];
        kdTree->annkSearch(
                point,             // the query point
                1,                 // number of near neighbors to return
                neighbourindex,    // nearest neighbor indices (returned)
                neighbourdistance, // the approximate nearest neighbor
                0);                // the error bound
        Normals.row(i)=normal.row(*neighbourindex);
        MatrixA.row(i)=VB.row(i);
        MatrixB.row(i)=VA.row(*neighbourindex);

    }

    Eigen::Matrix3d RotationMatrix;
    Eigen::Vector3d TranslationMatrix;

    Eigen::MatrixXd A,B;
    A.setZero(minnum,6);
    B.setZero(minnum,1);
    for (int i=0; i<minnum; i++) {
        double n1,n2,n3,a1,a2,a3,b1,b2,b3;
        a1=MatrixA(i,0);
        a2=MatrixA(i,1);
        a3=MatrixA(i,2);
        b1=MatrixB(i,0);
        b2=MatrixB(i,1);
        b3=MatrixB(i,2);
        n1=Normals(i,0);
        n2=Normals(i,1);
        n3=Normals(i,2);
        A.row(i)<<n3*a2-a3*n2,n1*a3-a1*n3,n2*a1-n1*a2,Normals.row(i);
        B.row(i)<<n1*(b1-a1)+n2*(b2-a2)+n3*(b3-a3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeThinU|Eigen::ComputeThinV);
    svdU=svd.matrixU();
    svdV=svd.matrixV();
    Eigen::MatrixXd X;
    X.setZero(6,1);
    double alpha,beta,gamma;
    double acos,asin,bcos,bsin,gcos,gsin;
    acos=cos(alpha);
    asin=sin(alpha);
    bcos=cos(beta);
    bsin=sin(beta);
    gcos=cos(gamma);
    gsin=sin(gamma);
    RotationMatrix <<
            bcos*gcos,-gsin*acos+gcos*bsin*asin,gsin*asin+gcos*bsin*acos,
            gsin*bcos,gcos*acos+asin*bsin*gsin,-gcos*asin+gsin*bsin*acos,
            -bsin,bcos*asin,bcos*acos;

    Eigen::MatrixXd V;
    V = (RotationMatrix*VB.transpose()).transpose()+TranslationMatrix.replicate(1,numB).transpose();
    return V;
}

Eigen::VectorXd Function::meanCurvature(Eigen::MatrixXd V1,Eigen::MatrixXi F1,Eigen::MatrixXd Vnormal,acq::NeighboursT Neighbours){
    // get the neighbors
    acq::NeighboursT neighbours=Neighbours;
    int count=V1.rows();
    // build laplacian matrix
    Eigen::SparseMatrix<double> Lap;
    Lap.resize(count,count);
    for(int i=0;i<count;i++){
        set<size_t> Neighind= Neighbours.at(i);
        set<size_t>::iterator it;
        double sum= Neighind.size();
        // diagonal value
        Lap.insert(i,i)= -sum;
        for(it = Neighind.begin();it!= Neighind.end();it++){
            // Lap (i,j)
            Lap.insert(i,*it)=1.0;
        }
    }
    Eigen::MatrixXd Vec=Lap*V1;
    Eigen::VectorXd H=Vec.rowwise().norm()/2;
    for(int i= 0;i< count;i++) {
        if (Vec.row(i).dot(Vnormal.row(i))>0) {
            H.row(i)*= -1;
        }
    }
    return H;
}

Eigen::VectorXd Function::gaussianCurvature(Eigen::MatrixXd V1,Eigen::MatrixXi F1) {
    int count=V1.rows();
    int Fcount=F1.rows();
    Eigen::VectorXd K;
    K.resize(count);
    for (int i= 0; i< count; i++) {
        Eigen::RowVector3d pa,pb,pc;
        double SumAngle=0,SumArea=0;
        for (int j= 0; j< Fcount; j++) {
            Eigen::RowVector3i myFace= F1.row(j);
            int myNeigh=0;
            for(int m=0;m<3;m++){
                if(myFace(m)==i){
                    // find the three points in connected face
                    pa= V1.row(myFace((m)%3));
                    pb= V1.row(myFace((m+1)%3));
                    pc= V1.row(myFace((m+2)%3));
                    myNeigh++;
                }
            }
            double a,b,c,p;
            // the length of each edge in that face
            a=(pc-pb).norm();
            b=(pa-pc).norm();
            c=(pa-pb).norm();
            p=(a+b+c)/2;
            if (myNeigh>0) {
                // angle a
                double Angle_a=acos((b*b+c*c-a*a)/(2*c*b));
                //double area=sqrt(p*(p-a)*(p-b)*(p-c));
                double area=b*c*sin(Angle_a);
                // the whole triangulation area
                SumArea=SumArea+area;
                SumAngle=SumAngle+Angle_a;
            }
        }
        // the bule triangulation area
        SumArea=SumArea/3.0;
        SumAngle=2.0*M_PI-SumAngle;
        K(i) =SumAngle/SumArea;
    }
    return K;
}

Eigen::VectorXd Function::NonuniformCurvature(Eigen::MatrixXd V1,Eigen::MatrixXi F1,Eigen::MatrixXd Vnormal) {
    int count=V1.rows();
    int Fcount=F1.rows();
    Eigen::MatrixXi TT, TTi;
    // get the adjacency face
    igl::triangle_triangle_adjacency(F1,TT,TTi);
    // build sparse matrix L=M*C
    Eigen::SparseMatrix<double> Lap,M,C;
    Lap.resize(count,count);
    M.resize(count,count);
    C.resize(count,count);
    // for each vertex
    for (int i= 0;i< count;i++) {
        double SumC = 0;
        Eigen::RowVector3d pa,pb,pc;
        double SumAngle=0,SumArea=0;
        double a,b,c,p,Angle_a,alpha,beta,area,weightC;
        for (int j= 0; j< Fcount; j++) {
            Eigen::RowVector3i myFace= F1.row(j);
            int myNeigh=0;
            int edgeInd;
            // find the three points in connected face
            for(int m=0;m<3;m++){
                if(myFace(m)==i){
                    pa= V1.row(myFace((m)%3));
                    pb= V1.row(myFace((m+1)%3));
                    pc= V1.row(myFace((m+2)%3));
                    myNeigh++;
                    // the index of vj
                    edgeInd=(m+2)%3;
                }
            }
            // the length of each edge in that face
            a=(pc-pb).norm();
            b=(pa-pc).norm();
            c=(pa-pb).norm();
            p=(a+b+c)/2;
            if (myNeigh>0) {
                // calculate the angle A
                Angle_a=acos((b*b+c*c-a*a)/(2*c*b));
                // anlge beta
                beta =acos((a*a-b*b+c*c)/(2*c*a));
                //double area=sqrt(p*(p-a)*(p-b)*(p-c));
                area=b*c*sin(Angle_a)/2;
                SumArea=SumArea+area;
                // get the index of adjacent face
                int indTrian = TT(j,edgeInd); //Index of the triangle adjacent to the j edge of triangle i.
                Eigen::RowVector3i NeighF= F1.row(indTrian);
                int indEdge= TTi(j,edgeInd);  //Index of edge of the triangle TT
                for (int m=0;m<3;m++){
                    // find the three points in adjacent face
                    if(m==indEdge){
                        pa= V1.row(NeighF((m)%3));
                        pb= V1.row(NeighF((m+1)%3));
                        pc= V1.row(NeighF((m+2)%3));
                    }
                }
                // the length of each edge in that face
                a=(pc-pb).norm();
                b=(pa-pc).norm();
                c=(pa-pb).norm();
                // angle alpha in the adjacent face
                alpha= acos((a*a+b*b-c*c)/(2*a*b));
                // value in Cij
                weightC=1/tan(alpha)+1/tan(beta);
                C.insert(i,myFace(edgeInd))= weightC;
                SumC=SumC-weightC;
            }
        }
        M.insert(i,i)=3/(2*SumArea);
        C.insert(i,i)=SumC; //diagonal
    }
    Lap= M*C;
    Eigen::MatrixXd Vec=Lap*V1;
    // get the absolute value
    Eigen::VectorXd H=Vec.rowwise().norm()/2;
    // define the orientation
    for(int i= 0;i< count;i++) {
        if (Vec.row(i).dot(Vnormal.row(i))>0) {
            H.row(i)*= -1;
        }
    }
    return H;
}

Eigen::MatrixXd Function::ReConstruction(Eigen::MatrixXd V1,Eigen::MatrixXi F1,int k){
    int count=V1.rows();
    int Fcount=F1.rows();
    Eigen::MatrixXi TT, TTi;
    // get the adjacency face
    igl::triangle_triangle_adjacency(F1,TT,TTi);
    // build sparse matrix L=M*C
    Eigen::SparseMatrix<double> Lap,M,C;
    Lap.resize(count,count);
    M.resize(count,count);
    C.resize(count,count);
    // for each vertex
    for (int i= 0;i< count;i++) {
        double SumC = 0;
        Eigen::RowVector3d pa,pb,pc;
        double SumAngle=0,SumArea=0;
        double a,b,c,p,Angle_a,alpha,beta,area,weightC;
        for (int j= 0; j< Fcount; j++) {
            Eigen::RowVector3i myFace= F1.row(j);
            int myNeigh=0;
            int edgeInd;
            // find the three points in connected face
            for(int m=0;m<3;m++){
                if(myFace(m)==i){
                    pa= V1.row(myFace((m)%3));
                    pb= V1.row(myFace((m+1)%3));
                    pc= V1.row(myFace((m+2)%3));
                    myNeigh++;
                    // the index of vj
                    edgeInd=(m+2)%3;
                }
            }
            // the length of each edge in that face
            a=(pc-pb).norm();
            b=(pa-pc).norm();
            c=(pa-pb).norm();
            p=(a+b+c)/2.0;
            if (myNeigh>0) {
                // calculate the angle A
                Angle_a=acos((b*b+c*c-a*a)/(2.0*c*b));
                // anlge beta
                beta =acos((a*a-b*b+c*c)/(2.0*c*a));
                //double area=sqrt(p*(p-a)*(p-b)*(p-c));
                area=b*c*sin(Angle_a)/2.0;
                SumArea=SumArea+area;
                // get the index of adjacent face
                int indTrian = TT(j,edgeInd); //Index of the triangle adjacent to the j edge of triangle i.
                Eigen::RowVector3i NeighF= F1.row(indTrian);
                int indEdge= TTi(j,edgeInd);  //Index of edge of the triangle TT
                for (int m=0;m<3;m++){
                    // find the three points in adjacent face
                    if(m==indEdge){
                        pa= V1.row(NeighF((m)%3));
                        pb= V1.row(NeighF((m+1)%3));
                        pc= V1.row(NeighF((m+2)%3));
                    }
                }
                // the length of each edge in that face
                a=(pc-pb).norm();
                b=(pa-pc).norm();
                c=(pa-pb).norm();
                // angle alpha in the adjacent face
                alpha= acos((a*a+b*b-c*c)/(2.0*a*b));
                // value in Cij
                weightC=1.0/tan(alpha)+1.0/tan(beta);
                C.insert(i,myFace(edgeInd))= weightC;
                SumC=SumC-weightC;
            }
        }
        M.insert(i,i)=3.0/(2.0*SumArea);
        C.insert(i,i)=SumC; //diagonal
    }
    Lap= M*C;

    typedef Eigen::SparseMatrix<double> SparseMat;
    typedef Eigen::SimplicialLDLT<SparseMat> SparseChol;
    typedef Eigen::ArpackGeneralizedSelfAdjointEigenSolver <SparseMat, SparseChol> Arpack;
    Arpack arpack;
    // define sparse matrix A
    SparseMat A;
    //calculate the k smallest eigenvalues
    int nbrEigenvalues =k;
    //arpack.compute(L, nbrEigenvalues, "SM");
    //cout << "arpack eigenvalues\n" << arpack.eigenvalues().transpose() << endl;
    //Spectra::GenEigsSolver
    Eigen::MatrixXd V;
    Eigen::VectorXd D;
    // igl::eigs(Lap,Lap,k,igl::EIGS_TYPE_SM,V,D);

    Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(Lap);
    // get the eigenvectors after sorting
    Eigen::MatrixXd EigenV=eigenSolver.eigenvectors();
    // x,y,z
    Eigen::MatrixXd x=V1.col(0);
    Eigen::MatrixXd y=V1.col(1);
    Eigen::MatrixXd z=V1.col(2);
    Eigen::MatrixXd newV(count,3);
    newV.setZero();
    Eigen::MatrixXd newX=newV.col(0);
    Eigen::MatrixXd newY=newV.col(1);
    Eigen::MatrixXd newZ=newV.col(2);
    MatrixXd ee(V1.rows(),k);
    ee.setZero();
    // get the first k smallest eigenvectors
   for(int i=0;i<k;i++){
       ee.col(i)=EigenV.col(count-i);
   }
    // update x,y,z
    newX=ee*(x.transpose()*ee).transpose();
    newY=ee*(y.transpose()*ee).transpose();
    newZ=ee*(z.transpose()*ee).transpose();
    newV<< newX, newY, newZ;
    return newV;
}

Eigen::MatrixXd Function::exSmoothing(Eigen::MatrixXd V1,Eigen::MatrixXi F1,acq::NeighboursT Neighbours,
                                      double lambda,int type,int iteration) {

    int count = V1.rows();
    int Fcount = F1.rows();
    acq::NeighboursT neighbours=Neighbours;
    Eigen::MatrixXi TT, TTi;
    // get the adjacency face
    igl::triangle_triangle_adjacency(F1,TT,TTi);
    // build sparse matrix L=M*C
    Eigen::SparseMatrix<double> Lap,M,C;
    Lap.resize(count,count);
    M.resize(count,count);
    C.resize(count,count);
    if (type==1){
        for(int i=0;i<count;i++){
            set<size_t> Neighind= Neighbours.at(i);
            set<size_t>::iterator it;
            double sum= Neighind.size();
            Lap.insert(i,i)= -sum;
            for(it = Neighind.begin();it!= Neighind.end();it++){
                Lap.insert(i,*it)=1;
            }
        }}
    else{ // for each vertex
        for (int i= 0;i< count;i++) {
            double SumC = 0;
            Eigen::RowVector3d pa,pb,pc;
            double SumAngle=0,SumArea=0;
            double a,b,c,p,Angle_a,alpha,beta,area,weightC;
            for (int j= 0; j< Fcount; j++) {
                Eigen::RowVector3i myFace= F1.row(j);
                int myNeigh=0;
                int edgeInd;
                // find the three points in connected face
                for(int m=0;m<3;m++){
                    if(myFace(m)==i){
                        pa= V1.row(myFace((m)%3));
                        pb= V1.row(myFace((m+1)%3));
                        pc= V1.row(myFace((m+2)%3));
                        myNeigh++;
                        // the index of vj
                        edgeInd=(m+2)%3;
                    }
                }
                // the length of each edge in that face
                a=(pc-pb).norm();
                b=(pa-pc).norm();
                c=(pa-pb).norm();
                p=(a+b+c)/2;
                if (myNeigh>0) {
                    // calculate the angle A
                    Angle_a=acos((b*b+c*c-a*a)/(2.0*c*b));
                    // anlge beta
                    beta =acos((a*a-b*b+c*c)/(2.0*c*a));
                    //double area=sqrt(p*(p-a)*(p-b)*(p-c));
                    area=b*c*sin(Angle_a)/2.0;
                    SumArea=SumArea+area;
                    // get the index of adjacent face
                    int indTrian = TT(j,edgeInd); //Index of the triangle adjacent to the j edge of triangle i.
                    Eigen::RowVector3i NeighF= F1.row(indTrian);
                    int indEdge= TTi(j,edgeInd);  //Index of edge of the triangle TT
                    for (int m=0;m<3;m++){
                        // find the three points in adjacent face
                        if(m==indEdge){
                            pa= V1.row(NeighF((m)%3));
                            pb= V1.row(NeighF((m+1)%3));
                            pc= V1.row(NeighF((m+2)%3));
                        }
                    }
                    // the length of each edge in that face
                    a=(pc-pb).norm();
                    b=(pa-pc).norm();
                    c=(pa-pb).norm();
                    // angle alpha in the adjacent face
                    alpha= acos((a*a+b*b-c*c)/(2.0*a*b));
                    // value in Cij
                    weightC=1.0/tan(alpha)+1.0/tan(beta);
                    C.insert(i,myFace(edgeInd))= weightC;
                    SumC=SumC-weightC;
                }
            }
            M.insert(i,i)=3.0/(2.0*SumArea);
            C.insert(i,i)=SumC; //diagonal
        }
        Lap= M*C;
}
    // update the vertex by explicit smoothing
    Eigen::SparseMatrix<double> eye(count, count);
    eye.setIdentity();
    Eigen::MatrixXd newV;
    // P(t+1)=(I+lambda*Lapla)
    newV= (eye+lambda*Lap) * V1;
    for (int i = 0; i <iteration; i++) {
        newV= (eye+lambda*Lap) * newV;
    }
    return newV;
}

Eigen::MatrixXd Function::imSmoothing(Eigen::MatrixXd V1,Eigen::MatrixXi F1,
                                      double lambda,int iteration){
    int count = V1.rows();
    int Fcount = F1.rows();
    // get the adjacency face
    Eigen::MatrixXi TT, TTi;
    igl::triangle_triangle_adjacency(F1, TT, TTi);
    const int n = V1.rows();
    // build sparse matrix L=M*C
    Eigen::SparseMatrix<double> Lap,C,Mm,M;
    Lap.resize(count,count);
    M.resize(count,count);
    Mm.resize(count,count);
    C.resize(count,count);
    // for each vertex
    for (int i = 0; i < count; i++) {
        double SumC = 0;
        Eigen::RowVector3d pa, pb, pc;
        double SumAngle = 0, SumArea = 0;
        double a, b, c, p, Angle_a, alpha, beta, area, weightC;
        for (int j = 0; j < Fcount; j++) {
            Eigen::RowVector3i myFace = F1.row(j);
            int myNeigh = 0;
            int edgeInd;
            // find the three points in connected face
            for (int m = 0; m < 3; m++) {
                if (myFace(m) == i) {
                    pa = V1.row(myFace((m) % 3));
                    pb = V1.row(myFace((m + 1) % 3));
                    pc = V1.row(myFace((m + 2) % 3));
                    myNeigh++;
                    // the index of vj
                    edgeInd = (m + 2) % 3;
                }
            }
            // the length of each edge in that face
            a = (pc - pb).norm();
            b = (pa - pc).norm();
            c = (pa - pb).norm();
            p = (a + b + c) / 2;
            if (myNeigh > 0) {
                // calculate the angle A
                Angle_a=acos((b*b+c*c-a*a)/(2.0*c*b));
                // anlge beta
                beta =acos((a*a-b*b+c*c)/(2.0*c*a));
                //double area=sqrt(p*(p-a)*(p-b)*(p-c));
                area = b * c * sin(Angle_a)/2;
                SumArea = SumArea + area;
                // get the index of adjacent face
                int indTrian = TT(j,edgeInd); //Index of the triangle adjacent to the j edge of triangle i.
                Eigen::RowVector3i NeighF= F1.row(indTrian);
                int indEdge= TTi(j,edgeInd);  //Index of edge of the triangle TT
                for (int m = 0; m < 3; m++) {
                    // find the three points in adjacent face
                    if (m == indEdge) {
                        pa = V1.row(NeighF((m) % 3));
                        pb = V1.row(NeighF((m + 1) % 3));
                        pc = V1.row(NeighF((m + 2) % 3));
                    }
                }
                // the length of each edge in that face
                a=(pc-pb).norm();
                b=(pa-pc).norm();
                c=(pa-pb).norm();
                // angle alpha in the adjacent face
                alpha = acos((pb - pc).dot(pa - pc) / (a*b));
                weightC = 1.0/ tan(alpha) + 1.0/ tan(beta);
                // value in Cij
                C.insert(i, myFace(edgeInd)) = weightC;
                SumC = SumC - weightC;
            }
        }
        C.insert(i, i) = SumC;
        M.insert(i, i) = 2.0 * (SumArea/3.0);
        // Mm is the inverse of M
        Mm.insert(i, i) = 3.0/ (2.0*SumArea);
    }
    Lap = Mm * C;
    Eigen::MatrixXd newV;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    // Lw= M*L
    solver.compute(M - lambda * M * Lap);
    // (M-lambda*Lw)*P(t+1)=M*P(t)
    Eigen::MatrixXd x=solver.solve(M * V1);
    for(int i=0;i<iteration;i++){
        // the right is M*P(t)
        x=solver.solve(M * x);
    }
    return x;
}




