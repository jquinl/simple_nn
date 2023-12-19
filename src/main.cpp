#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sqlite3.h>


#define DEBUG_SHAPE(x) std::cout<< "("<< x.rows()<<","<<x.cols()<<")"<<"("<< x.size()<<")"<< std::endl
#define DEBUG_SHOW_MATRIX(x) std::cout << "(" << x << ")" << "("<< x.rows()<<","<<x.cols()<<")"<<"("<< x.size()<<")"<< std::endl
#define DEBUG_PRINT(x) std::cout << x << std::endl

//Activation function struct
typedef Eigen::MatrixXf (*actfn_ptr)(const Eigen::MatrixXf &m1);
struct Afunc
{
    actfn_ptr fn;
    actfn_ptr dfn;
};
struct Layer{
    int in_size;
    int l_size;
    Afunc act;
    Eigen::MatrixXf w;
    Eigen::VectorXf b;

    Layer(int input_size, int layer_size, Afunc activation_function)
        :in_size(input_size),
        l_size(layer_size),
        act(activation_function),
        w(Eigen::MatrixXf::Random(layer_size,input_size) * 0.5f),
        b(Eigen::VectorXf::Random(layer_size) *0.5f)
        {
    }

    Eigen::MatrixXf ForwardPropagate(const Eigen::MatrixXf& input){
        Eigen::MatrixXf it = (w * input).colwise() + b;
        return act.fn(it);
    }
};
Eigen::MatrixXf OneHot( const Eigen::VectorXi& labels ){
    Eigen::MatrixXf oneh = Eigen::MatrixXf::Zero(labels.size(),labels.maxCoeff()+1);
    //Ctm que feo esto
    for(int i=0;i<labels.size();i++){
        oneh(i,labels(i)) = 1;
    }
    return oneh.transpose();
}
class NNCateg{
    float alpha;
    float n_out;
    std::vector<Layer*> layers;
public:
    NNCateg(std::vector<Layer*> layers,float alpha,int n_output)
        :layers(layers),alpha(alpha),n_out(float(n_output)){}

    void Train(const Eigen::MatrixXf& train_data_x,const Eigen::MatrixXf& train_data_y ){
        //assign continous block of heap to all the outputs
        int l_sz = layers.size();
        int n_data = train_data_x.cols();

        int t_size = layers[0]->l_size * n_data;
        for (int i = 1; i< l_sz; i++){
            t_size+= layers[i]->l_size * n_data;
        }

        float* l_out = new float[t_size];

        //Forward Prop
        {
            Eigen::Map<Eigen::MatrixXf> a_layer(
                l_out  , layers[0]->l_size, n_data
            );
            a_layer << layers[0]->ForwardPropagate(train_data_x);
        }
        int adv = layers[0]->l_size * n_data;
        for (int i = 1; i< l_sz; i++){

            Eigen::Map<Eigen::MatrixXf> in_l (
                l_out + adv - (layers[i-1]->l_size * n_data), layers[i-1]->l_size, n_data
            );
            Eigen::Map<Eigen::MatrixXf> a_layer(
                l_out + adv , layers[i]->l_size, n_data
            );

            a_layer << layers[i]->ForwardPropagate(in_l);

            adv += layers[i]->l_size *  n_data;
        }

        //back Prop
        int dw_size,db_size,dz_size = 0;

        for (int i = 1; i< l_sz; i++){
            dw_size += layers[i]->l_size * layers[i]->in_size;
            db_size += layers[i]->l_size;
            dz_size = dz_size < (layers[i]->l_size * n_data) ? (layers[i]->l_size * n_data) : dz_size;
        }

        float* dws = new float[dw_size];
        float* dbs = new float[db_size];
        float* dzs = new float[dz_size];

        float* dws_p = dws + dw_size;
        float* dbs_p = dbs + db_size;
        {
            // Shifting pointers to map w,b and a to the arrays
            dws_p -= layers[l_sz-1]->l_size * layers[l_sz-1]->in_size;
            dbs_p -= layers[l_sz-1]->l_size;

            Eigen::Map<Eigen::MatrixXf> in_l (
                l_out + adv - (layers[l_sz-2]->l_size * n_data), layers[l_sz-2]->l_size, n_data
            );
            Eigen::Map<Eigen::MatrixXf> a_layer(
                l_out + adv , layers[l_sz-1]->l_size, n_data
            );
            Eigen::Map<Eigen::MatrixXf> dz_l(
                dzs , layers[l_sz-1]->l_size, n_data
            );
            Eigen::Map<Eigen::MatrixXf> dw_l (
                dws_p , layers[l_sz-1]->l_size,layers[l_sz-1]->in_size
            );
            Eigen::Map<Eigen::VectorXf> db_l (
                dbs_p , layers[l_sz-1]->l_size
            );

            adv -= layers[l_sz-1]->l_size *  n_data;
            //Here backprop starts proper
            dz_l << a_layer - train_data_y;
            DEBUG_SHAPE((1.f/n_out * (dz_l * in_l.transpose())));
            DEBUG_SHAPE(dw_l);
            DEBUG_SHAPE((1.f/n_out * dz_l.colwise().sum()));
            DEBUG_SHAPE(db_l);
            dw_l << 1.f/n_out * (dz_l * in_l.transpose()) ;
        //    db_l << 1.f/n_out * dz_l.colwise().sum() ;
        }
        for (int i = l_sz-2; i > 0; i--){
            dws_p -= layers[i]->l_size * layers[i]->in_size;
            dbs_p -= layers[i]->l_size;
            Eigen::Map<Eigen::MatrixXf> in_l (
                l_out + adv - (layers[i-1]->l_size * n_data), layers[i-1]->l_size, n_data
            );
            Eigen::Map<Eigen::MatrixXf> a_layer(
                l_out + adv, layers[i]->l_size, n_data
            );
            Eigen::Map<Eigen::MatrixXf> dz_l(
                dzs, layers[i+1]->l_size, n_data
            );
            Eigen::Map<Eigen::MatrixXf> dz_o(
                dzs, layers[i]->l_size, n_data
            );
            Eigen::Map<Eigen::MatrixXf> dw_l(
                dws_p, layers[i]->l_size,layers[i]->in_size
            );
            Eigen::Map<Eigen::VectorXf> db_l(
                dbs_p, layers[i]->l_size
            );

            adv -= layers[i]->l_size *  n_data;

            //IDk if i can do this
        //    dz_o << (layers[i]->w.transpose() * dz_l).cwiseProduct(layers[i-1]->act.dfn(in_l)) ;

        //    dw_l <<  1.f/n_out * (dz_o * in_l.transpose()) ;
         //   db_l << 1.f/n_out * dz_o.colwise().sum() ;
        }
        //Output layer missing
        delete[] dzs;
        delete[] dbs;
        delete[] dws;
        delete[] l_out;
    }


private:
    Eigen::MatrixXf BackPropagateOut(const Eigen::MatrixXf& a_c , const Eigen::MatrixXf& out_onehot){
        return a_c - out_onehot;
    }
    Eigen::MatrixXf BackPropagateDz(const Layer& c_layer ,const Layer& n_layer, const Eigen::MatrixXf& a_c, const Eigen::MatrixXf& dz_next){
        return (n_layer.w.transpose() * dz_next).cwiseProduct(c_layer.act.dfn(a_c)) ;
    }
    Eigen::MatrixXf BackPropagateWB(const Layer& p_layer,Eigen::MatrixXf& dz){
     //   Eigen::MatrixXf dw = 1.f/n_out * (dz * p_layer.a.transpose());
        Eigen::VectorXf db = 1.f/n_out * dz.colwise().sum();
        //Change this
        return db;
    }
};
int main(int argc, char *argv[]){

    //Input layer 
    int input_vector_size = 5;//<--- size of a single datapoint
    int training_points = 10;//<--- Number of training poìnts

    Eigen::VectorXi label_data(training_points);
    label_data <<  1,3,4,3,5,4,2,1,4,2;

    //Layer sizes
    int first_layer_size  = 10;//<--- Size of the first layer
    int second_layer_size = 20;//<--- Size of the first layer
    int output_layer_size = label_data.maxCoeff()+1;//<---last layer output must coincide with data classes + 1

    actfn_ptr softmax   =  *[](const  Eigen::MatrixXf &m) {return Eigen::MatrixXf( m.array().exp().rowwise() / m.array().exp().colwise().sum());};
  //actfn_ptr softmax_d =  *[]( const Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.exp()/m.exp().sum());};

    actfn_ptr sigmoid   =  *[](const  Eigen::MatrixXf &m) {return Eigen::MatrixXf(1.0f/(1.0f + m.array().exp() ));};
    actfn_ptr sigmoid_d =  *[]( const Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.array() * ((1.0f - m.array())));};

    actfn_ptr relu   =  *[](const  Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.cwiseMax(0));};
    actfn_ptr relu_d =  *[](const Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.cwiseGreater(0).cast<float>());};

    Afunc activ_softmax = Afunc{
                    softmax,
                    softmax
                };
    Afunc activ_sigmoid = Afunc{
                    sigmoid,
                    sigmoid_d
                };
    Afunc activ_relu = Afunc{
                    relu,
                    relu_d
                };

    Eigen::MatrixXf x = Eigen::MatrixXf::Random(training_points,input_vector_size).transpose();

    Layer l1_cls(input_vector_size,first_layer_size,activ_relu);
    Layer l2_cls(first_layer_size,second_layer_size,activ_sigmoid);
    Layer l3_cls(second_layer_size,output_layer_size,activ_softmax);

    std::vector<Layer*> layers{&l1_cls,&l2_cls,&l3_cls};

    Eigen::MatrixXf a1 = l1_cls.ForwardPropagate(x);
    Eigen::MatrixXf a2 = l2_cls.ForwardPropagate(a1);
    Eigen::MatrixXf a3 = l3_cls.ForwardPropagate(a2);

    Eigen::MatrixXf onehot = OneHot(label_data);
    
    NNCateg nn(layers,0.1f,10.0f);
    nn.Train(x,onehot);

    //Backprop
    float t = float(training_points);

    Eigen::MatrixXf dz3 = a3 - onehot;
    Eigen::MatrixXf dw3 = 1.f/t * (dz3 * a2.transpose()) ;
    Eigen::MatrixXf db3 = 1.f/t * dz3.colwise().sum() ;

    // dx = nextlayer_w, next layer err, self_layer deriv self_layer output
    Eigen::MatrixXf dz2 = (l3_cls.w.transpose() * dz3).cwiseProduct(l2_cls.act.dfn(a2)) ;
    DEBUG_SHAPE(a3);
    DEBUG_SHAPE(dz3);
    // //w= prevlayer_output
    Eigen::MatrixXf dw2 = 1.f/t * (dz2 * a1.transpose()) ;
    Eigen::MatrixXf db2 = 1.f/t * dz2.colwise().sum() ;
    DEBUG_SHAPE(a2);
    DEBUG_SHAPE(dz2);
    Eigen::MatrixXf dz1 = (l2_cls.w.transpose() * dz2).cwiseProduct(l1_cls.act.dfn(a1)) ;
    DEBUG_SHAPE(a1);
    DEBUG_SHAPE(dz1);
     DEBUG_SHAPE(db2);
    DEBUG_SHAPE(l2_cls.b);
    Eigen::MatrixXf dw1 = 1.f/t * (dz1 * x.transpose()) ;
    Eigen::MatrixXf db1 = 1.f/t * dz1.colwise().sum();
    DEBUG_SHAPE(db1);
    DEBUG_SHAPE(l1_cls.b);
    //Update Params

    return 0;
}
