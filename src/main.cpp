#include <iostream>
#include <Eigen/Dense>
#include <sqlite3.h>


#define DEBUG_SHAPE(x) std::cout<< "("<< x.rows()<<","<<x.cols()<<")"<< std::endl
#define DEBUG_PRINT(x) std::cout << "(" << x << ")" << std::endl

//Activation function struct
typedef Eigen::MatrixXf (*actfn_ptr)(Eigen::MatrixXf &m1);
struct Afunc
{
    actfn_ptr act_fn;
    actfn_ptr act_dfn;
};
struct Layer{

    int in_size;
    int l_size;
    Afunc a_fn;
    Eigen::MatrixXf w;
    Eigen::VectorXf b;

    Layer(int input_size, int layer_size, Afunc activation_function)
        :in_size(input_size),
        l_size(layer_size),
        a_fn(activation_function),
        w(Eigen::MatrixXf::Random(layer_size,input_size) * 0.5f),
        b(Eigen::VectorXf::Random(layer_size) *0.5f){
    }

    Eigen::MatrixXf ForwardPropagate(const Eigen::MatrixXf& input){
        Eigen::MatrixXf it = (w * input.transpose()).colwise() + b;
        return a_fn.act_fn(it);
    }
    void BackPropagate(){

    }
};
int main(int argc, char *argv[]){

    //Input layer 
    int input_vector_size = 25;//<--- size of a single datapoint
    int input_layer_size = 10;//<--- Number of training poìnts

    //first layer 
    int first_layer_size = 10;//<--- Size of the first layer
    int second_layer_size = 20;//<--- Size of the first layer


    actfn_ptr sigmoid   =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(1.0f/(1.0f + m.array().exp()));};
    actfn_ptr sigmoid_d =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.array() * ((1.0f - m.array())));};

    actfn_ptr relu   =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.cwiseMax(0));};
    //actfn_ptr relu_d =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.cwiseGreater(0));};
    Afunc activ_sigmoid = Afunc{
                    sigmoid,
                    sigmoid_d
                };
    Afunc activ_relu = Afunc{
                    relu,
                    relu
                };
    //Input 
    Eigen::MatrixXf x = Eigen::MatrixXf::Random(input_layer_size,input_vector_size);

    Layer l1_cls(input_vector_size,input_layer_size,activ_sigmoid);
    Layer l2_cls(input_layer_size,second_layer_size,activ_relu);

    Eigen::MatrixXf m2 = l2_cls.ForwardPropagate(l1_cls.ForwardPropagate(x));

    DEBUG_PRINT(m2);
    return 0;
}