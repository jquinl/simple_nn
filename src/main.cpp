#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>
#include <sstream>
#include <iostream>

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
    const Eigen::MatrixXf ForwardPropagate(const Eigen::MatrixXf& input){
        Eigen::MatrixXf it = (w * input).colwise() + b;
        return act.fn(it);
    }
    void UpdateParams(const Eigen::MatrixXf& dw,const Eigen::VectorXf& db,float alpha){
        w -= dw * alpha;
        b -= db * alpha;
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

    void Train(const Eigen::MatrixXf& train_data_x,const Eigen::MatrixXf& train_data_y,int steps ){
    
        //assign continous block of heap to all the outputs
        int l_sz = layers.size();
        int n_data = train_data_x.cols();

        int t_size,dw_size,db_size,dz_size = 0;

        for (int i = 0; i< l_sz; i++){
            t_size+= layers[i]->l_size * n_data;
            dw_size += layers[i]->l_size * layers[i]->in_size;
            db_size += layers[i]->l_size;
            dz_size = dz_size < (layers[i]->l_size * n_data) ? (layers[i]->l_size * n_data) : dz_size;
        }

        float* dws = new float[dw_size];
        float* dbs = new float[db_size];
        float* dzs = new float[dz_size];
        float* l_out = new float[t_size];

        for (int s = 0; s< steps; s++){
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
                dz_l << a_layer - train_data_y;
                dw_l << 1.f/n_out * (dz_l * in_l.transpose()) ;
                db_l << 1.f/n_out * dz_l.rowwise().sum() ;
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

                dz_o << (layers[i+1]->w.transpose() * dz_l).cwiseProduct(layers[i]->act.dfn(a_layer)) ;
                dw_l <<  1.f/n_out * (dz_o * in_l.transpose()) ;
                db_l << 1.f/n_out * dz_o.rowwise().sum() ;
            }
            //Output layer missing
            {
                dws_p -= layers[0]->l_size * layers[0]->in_size;
                dbs_p -= layers[0]->l_size;

                Eigen::Map<Eigen::MatrixXf> a_layer(
                    l_out + adv, layers[0]->l_size, n_data
                );
                Eigen::Map<Eigen::MatrixXf> dz_l(
                    dzs, layers[1]->l_size, n_data
                );
                Eigen::Map<Eigen::MatrixXf> dz_o(
                    dzs, layers[0]->l_size, n_data
                );
                Eigen::Map<Eigen::MatrixXf> dw_l(
                    dws_p, layers[0]->l_size,layers[0]->in_size
                );
                Eigen::Map<Eigen::VectorXf> db_l(
                    dbs_p, layers[0]->l_size
                );

                //adv -= layers[0]->l_size *  n_data;
                dz_o << (layers[1]->w.transpose() * dz_l).cwiseProduct(layers[0]->act.dfn(a_layer)) ;

                dw_l << 1.f/n_out * (dz_o * train_data_x.transpose()) ;

                db_l << 1.f/n_out * dz_o.rowwise().sum() ;
            }
            //Update params
            for (int i = 0; i< l_sz; i++){
                Eigen::Map<Eigen::MatrixXf> dw_l(
                    dws_p, layers[i]->l_size,layers[i]->in_size
                );
                Eigen::Map<Eigen::VectorXf> db_l(
                    dbs_p, layers[i]->l_size
                );
                dws_p += layers[i]->l_size * layers[i]->in_size;
                dbs_p += layers[i]->l_size;

                layers[i]->UpdateParams(dw_l,db_l,alpha);
            }
        }

        delete[] l_out;
        delete[] dzs;
        delete[] dbs;
        delete[] dws;
    }

    Eigen::MatrixXf Predict(const Eigen::MatrixXf& prediction_data){
        return prediction_data;
    }
};
Eigen::MatrixXf CSVToMatrix(std::string filename){
    std::ifstream input(filename);
    if(!input.is_open()){
        std::cerr << "Could not read file: "<< filename << "\n";
        return Eigen::MatrixXf::Zero(1,1);
    }
    int input_vec_size = 0;
    int input_set_size = 1;
    {
        std::string line;
        std::getline(input, line);
        std::istringstream ss(std::move(line));
        for (std::string value; std::getline(ss, value, ',');) {
            input_vec_size++;
            }
        for ( line; std::getline(input, line);) {
            input_set_size++;
        }
    }
    //Go back to beggining to read and push to matrix
    input.clear();
    input.seekg(0, std::ios::beg);
    Eigen::MatrixXf ret = Eigen::MatrixXf::Zero(input_vec_size,input_set_size);
    float* ret_p = (float*) ret.data();

    float* in_vec = new float[input_vec_size];
    Eigen::Map<Eigen::VectorXf> in_vec_m(
                    in_vec, input_vec_size
                );
    int adv = 0;
    for (std::string line; std::getline(input, line);) {
        std::istringstream ss(std::move(line));
        float* in_p = in_vec;
        for (std::string value; std::getline(ss, value, ',');) {
            *in_p = std::stof(value);
            in_p++;
        }

        Eigen::Map<Eigen::VectorXf> ret_vec_m(
                    ret_p + input_vec_size * adv , input_vec_size
                );

        ret_vec_m << in_vec_m;

        adv++;
    }
    delete[] in_vec;
    return ret;
}
int main(int argc, char *argv[]){

    //Input layer
    Eigen::MatrixXf data = CSVToMatrix("mnist_train.csv");
    Eigen::MatrixXf x = data(Eigen::placeholders::all,1).transpose();
    Eigen::VectorXi label_data = Eigen::VectorXf(data(Eigen::placeholders::all,0)).cast<int>();

    int input_vector_size = x.rows();//<--- size of a single datapoint
    int training_points = x.cols();//<--- Number of training poìnts

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

    Layer l1_cls(input_vector_size,first_layer_size,activ_relu);
    Layer l2_cls(first_layer_size,second_layer_size,activ_sigmoid);
    Layer l3_cls(second_layer_size,output_layer_size,activ_softmax);

    std::vector<Layer*> layers{&l1_cls,&l2_cls,&l3_cls};

    Eigen::MatrixXf onehot = OneHot(label_data);

    NNCateg nn(layers,0.1f,10.0f);
    nn.Train(x,onehot,10);

    return 0;
}
