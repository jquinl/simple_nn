#include <iostream>
#include <Eigen/Dense>

int main(int argc, char *argv[]){

    Eigen::MatrixXd m(2,2);
    m(0,0)=1;
    std::cout << m<<std::endl;

    return 0;
}