#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <set>
#include <iomanip>
using namespace std;
using namespace std::chrono;

vector<vector<double>> data_matrix;
vector<char> class_vector;
random_device r;
// double Seed = r();
double Seed;

void readData(string file)
{
    vector<vector<string>> data_matrix_aux;
    string ifilename = file;
    ifstream ifile;
    istream *input = &ifile;

    ifile.open(ifilename.c_str());

    if (!ifile)
    {
        cerr << "[ERROR]Couldn't open the file" << endl;
        cerr << "[Ex.] Are you sure you are in the correct path?" << endl;
        exit(1);
    }

    string data;
    int cont = 0, cont_aux = 0;
    char aux;
    vector<string> aux_vector;
    bool finish = false;

    // Leo número de atributos y lo guardo en contador
    do
    {
        *input >> data;
        if (data == "@attribute")
            cont++;
    } while (data != "@data"); // A partir de aquí leemos datos

    data = "";

    // Mientras no lleguemos al final leemos datos
    while (!(*input).eof())
    {
        // Leemos caracter a caracter
        *input >> aux;

        /* Si hemos terminado una linea de datos la guardamos en la matrix de datos
        y reiniciamos el contador auxiliar (nos dice por qué dato vamos) */
        if (finish)
        {
            data_matrix_aux.push_back(aux_vector);
            aux_vector.clear();
            cont_aux = 0;
            finish = false;
        }

        /* Si hay una coma el dato ha terminado de leerse y lo almacenamos, en caso
        contrario seguimos leyendo caracteres y almacenandolos en data*/
        if (aux != ',' && cont_aux < cont)
        {
            data += aux;
            // Si hemos llegado al penultimo elemento hemos terminado
            if (cont_aux == cont - 1)
            {
                cont_aux++;
                aux_vector.push_back(data);
                data = "";
                finish = true;
            }
        }
        else
        {
            aux_vector.push_back(data);
            data = "";
            cont_aux++;
        }
    }

    vector<double> vect_aux;

    for (vector<vector<string>>::iterator it = data_matrix_aux.begin(); it != data_matrix_aux.end(); it++)
    {
        vect_aux.clear();
        for (vector<string>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            if (jt == it->end() - 1)
                class_vector.push_back((*jt)[0]);
            else
                vect_aux.push_back(stod(*jt));
        }
        data_matrix.push_back(vect_aux);
    }
}

void normalizeData(vector<vector<double>> &data)
{
    double item = 0.0;           // Característica individual
    double max_item = -999999.0; // Valor máximo del rango de valores
    double min_item = 999999.0;  // Valor minimo del rango de valores

    // Buscamos los máximos y mínimos
    for (vector<vector<double>>::iterator it = data.begin(); it != data.end(); it++)
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            item = *jt;

            if (item > max_item)
                max_item = item;

            if (item < min_item)
                min_item = item;
        }

    // Normalizamos aplicando x_iN = (x_i - min) / (max - min)
    for (vector<vector<double>>::iterator it = data.begin(); it != data.end(); it++)
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
            *jt = (*jt - min_item) / (max_item - min_item);
}

pair<vector<vector<vector<double>>>, vector<vector<char>>> createPartitions()
{
    vector<vector<double>> data_m_aux = data_matrix;
    vector<char> class_v_aux = class_vector;

    // Mezclo aleatoriamente la matriz original
    /*srand(Seed);
    random_shuffle(begin(data_m_aux), end(data_m_aux));
    srand(Seed);
    random_shuffle(begin(class_vector), end(class_vector));*/

    const int MATRIX_SIZE = data_matrix.size();
    vector<vector<double>>::iterator it = data_m_aux.begin();
    vector<char>::iterator jt = class_v_aux.begin();

    // Particiones y puntero que las irá recorriendolas para insertar datos
    vector<vector<double>> g1, g2, g3, g4, g5, *g_aux;
    vector<char> g1c, g2c, g3c, g4c, g5c, *g_aux2;
    int cont = 0, cont_grupos = 0;
    bool salir = false;

    // Mientras no se hayan insertado todos los datos en todos los grupos
    while (cont != MATRIX_SIZE && cont_grupos < 5)
    {
        // Elegimos la partición que toque
        switch (cont_grupos)
        {
        case 0:
            g_aux = &g1;
            g_aux2 = &g1c;
            break;
        case 1:
            g_aux = &g2;
            g_aux2 = &g2c;
            break;
        case 2:
            g_aux = &g3;
            g_aux2 = &g3c;
            break;
        case 3:
            g_aux = &g4;
            g_aux2 = &g4c;
            break;
        case 4:
            g_aux = &g5;
            g_aux2 = &g5c;
            break;
        }

        // Vamos rellenando la partición pertinente
        for (int k = 0; k < MATRIX_SIZE / 5 && !salir; k++)
        {
            g_aux->push_back(*it);
            g_aux2->push_back(*jt);
            it++;
            jt++;
            cont++;

            /* Si estamos en el último grupo y quedan todavía elementos, seguir
            insertándolos en este último */
            if (cont_grupos == 4)
            {
                if (it != data_m_aux.end())
                    k--;
                else
                    salir = true;
            }
        }
        cont_grupos++;
    }
    vector<vector<vector<double>>> d = {g1, g2, g3, g4, g5};
    vector<vector<char>> c = {g1c, g2c, g3c, g4c, g5c};
    pair<vector<vector<vector<double>>>, vector<vector<char>>> partitions = make_pair(d, c);

    return partitions;
}

char KNN_Classifier(vector<vector<double>> &data, vector<vector<double>>::iterator &elem, vector<char> &elemClass, vector<double> &w)
{
    vector<double> distancia;
    vector<char> clases;
    vector<char>::iterator cl = elemClass.begin();
    vector<double>::iterator wi = w.begin();
    vector<double>::iterator ej;
    double sumatoria = 0;
    double dist_e = 0;

    for (vector<vector<double>>::iterator e = data.begin(); e != data.end(); e++)
    {
        // Si el elemento es él mismo no calculamos distancia, pues es 0
        if (elem != e)
        {
            sumatoria = 0;
            ej = elem->begin();
            wi = w.begin();

            // Calculamos distancia de nuestro elemento con el resto
            for (vector<double>::iterator ei = e->begin(); ei != e->end(); ei++)
            {
                sumatoria += *wi * pow(*ej - *ei, 2);
                ej++;
                wi++;
            }
            dist_e = sqrt(sumatoria);
            distancia.push_back(dist_e);
            clases.push_back(*cl);
        }
        cl++;
    }

    vector<double>::iterator it;
    vector<char>::iterator cl_dist_min = clases.begin();

    double distMin = 99999;
    char vecinoMasProxClass;

    // Nos quedamos con el que tenga minima distancia, es decir, su vecino más próximo
    for (it = distancia.begin(); it != distancia.end(); it++)
    {
        if (*it < distMin)
        {
            distMin = *it;
            vecinoMasProxClass = *cl_dist_min;
        }
        cl_dist_min++;
    }

    return vecinoMasProxClass;
}

double calculaAciertos(vector<vector<double>> &muestras, vector<char> &clases, vector<double> &w)
{
    double instBienClasificadas = 0.0;
    double numIntanciasTotal = float(muestras.size());
    char cl_1NN;
    vector<char>::iterator c_it = clases.begin();

    for (vector<vector<double>>::iterator it = muestras.begin(); it != muestras.end(); it++)
    {
        cl_1NN = KNN_Classifier(muestras, it, clases, w);

        if (cl_1NN == *c_it)
            instBienClasificadas += 1.0;
        c_it++;
    }

    return instBienClasificadas / numIntanciasTotal;
}

void execute(pair<vector<vector<vector<double>>>, vector<vector<char>>> &part, vector<double> (*alg)(vector<vector<double>> &, vector<char> &))
{
    vector<double> w;
    vector<vector<vector<double>>>::iterator data_test = part.first.begin();
    vector<vector<char>>::iterator class_test = part.second.begin();
    vector<vector<double>> aux_data_fold;
    vector<char> aux_class_fold;
    vector<vector<vector<double>>>::iterator it;
    vector<vector<char>>::iterator jt;

    double tasa_clas = 0;
    double tasa_red = 0;
    double agregado = 0;
    double alpha = 0.5;
    unsigned int cont_red = 0;
    double TS_media = 0, TR_media = 0, A_media = 0;
    int cont = 0;

    auto momentoInicio = high_resolution_clock::now();

    // Iteramos 5 veces ejecutando el algoritmo
    while (cont < 5)
    {
        jt = part.second.begin();
        aux_data_fold.clear();
        aux_class_fold.clear();
        cont_red = 0;

        // Creamos particiones train
        for (it = part.first.begin(); it != part.first.end(); it++)
        {
            // Si es una partición test no la añadimos a training
            if (it != data_test && jt != class_test)
            {
                aux_data_fold.insert(aux_data_fold.end(), (*it).begin(), (*it).end());
                aux_class_fold.insert(aux_class_fold.end(), (*jt).begin(), (*jt).end());
            }
            jt++;
        }

        // Ejecución del algoritmo
        auto partInicio = high_resolution_clock::now();
        w = alg(aux_data_fold, aux_class_fold);
        auto partFin = high_resolution_clock::now();

        cont_red = 0;
        for (vector<double>::iterator wi = w.begin(); wi != w.end(); wi++)
        {
            if (*wi < 0.1)
            {
                cont_red += 1;
                *wi = 0.0;
            }
        }

        tasa_clas = calculaAciertos(*data_test, *class_test, w);
        tasa_red = float(cont_red) / float(w.size());
        agregado = alpha * tasa_clas + (1 - alpha) * tasa_red;

        milliseconds tiempo_part = duration_cast<std::chrono::milliseconds>(partFin - partInicio);

        std::cout << "[PART " << cont + 1 << "] | Tasa_clas: " << tasa_clas << endl;
        std::cout << "[PART " << cont + 1 << "] | Tasa_red: " << tasa_red << endl;
        std::cout << "[PART " << cont + 1 << "] | Fitness: " << agregado << endl;
        std::cout << "[PART " << cont + 1 << "] | Tiempo_ejecucion: " << tiempo_part.count() << " ms\n\n";
        std::cout << "-------------------------------------------\n"
                  << endl;

        TS_media += tasa_clas;
        TR_media += tasa_red;
        A_media += agregado;

        cont++;
        data_test++;
        class_test++;
    }
    auto momentoFin = high_resolution_clock::now();

    milliseconds tiempo = duration_cast<std::chrono::milliseconds>(momentoFin - momentoInicio);

    std::cout << "***** (RESULTADOS FINALES) *****\n"
              << endl;
    std::cout << "Tasa_clas_media: " << TS_media / 5.0 << endl;
    std::cout << "Tasa_red_media: " << TR_media / 5.0 << endl;
    std::cout << "Fitness_medio: " << A_media / 5.0 << endl;
    std::cout << "Tiempo_ejecucion_medio: " << tiempo.count() << " ms";
}

double evalua(vector<vector<double>> &muestra, vector<char> &muestra_clases, vector<double> &poblacion)
{
    double tasa_clas = 0;
    double tasa_red = 0;
    double cont_reducc = 0;
    double f;

    for (vector<double>::iterator it = poblacion.begin(); it != poblacion.end(); it++)
    {
        if (*it < 0.1)
        {
            cont_reducc += 1.0;
            *it = 0;
        }
    }
    tasa_clas = calculaAciertos(muestra, muestra_clases, poblacion);
    tasa_red = float(cont_reducc) / float(poblacion.size());
    f = tasa_red * 0.5 + tasa_clas * 0.5;

    return f;
}

vector<double> algEnfriamientoSimulado(vector<vector<double>> &muestra, vector<char> &clase)
{
    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    // Inizializo vector de pesos con el generador
    vector<double> w(muestra.begin()->size());
    vector<double> best_w(muestra.begin()->size());
    generate(begin(w), end(w), gen);

    // Vector de mutaciones
    vector<double> z(w.size());
    vector<double>::iterator z_it = z.begin();
    normal_distribution<double> normal_dist(0.0, sqrt(0.3));
    mt19937 other_eng(Seed);
    auto genNormalDist = [&normal_dist, &other_eng]()
    {
        return normal_dist(other_eng);
    };

    int max_vecinos = 10 * w.size();
    double dif = 0;
    int max_exitos = 0.1 * max_vecinos;
    int M = 15000 / max_vecinos;
    int cont_vecinos = 0;
    int cont_exitos = 1;
    double eval = 0, best_eval = 0, eval_aux = 0;
    double mu = 0.3;
    double k = 1.0;
    int iter = 0, index = 0;
    vector<double> w_aux;

    eval = evalua(muestra, clase, w);
    best_w.swap(w);
    best_eval = eval;

    // temperatura inicial
    double t_inicial = (mu * eval) / -log(mu);
    // temperatura final
    double t_final = 0.001;

    // temperatura final siempre menor que la inicial
    while (t_final > t_inicial)
        t_final *= 0.001;

    double beta = (t_inicial - t_final) / (M * t_final * t_inicial);

    // temperatura actual
    double t_actual = t_inicial;

    while (iter < M && cont_exitos > 0 && t_actual > t_final)
    {
        cont_exitos = 0;
        cont_vecinos = 0;

        generate(begin(z), end(z), genNormalDist);
        while (cont_exitos < max_exitos && cont_vecinos < max_vecinos)
        {
            w_aux = w;

            // comienzo mutaciones
            index = r() % w_aux.size();
            w_aux[index] += *z_it;
            z_it++;

            if (w_aux[index] < 0.0)
                w_aux[index] = 0.0;
            else if (w_aux[index] > 1.0)
                w_aux[index] = 1.0;

            // evaluo la nuevas soluciones
            eval_aux = evalua(muestra, clase, w_aux);
            dif = eval_aux - eval;

            if (dif > 0.0 || gen() <= (exp(dif) / (k * t_actual)))
            {
                eval = eval_aux;
                w.swap(w_aux);
                cont_exitos += 1;

                if (eval > best_eval)
                {
                    best_eval = eval;
                    best_w = w;
                }
            }

            cont_vecinos += 1;
        }

        t_actual = t_actual / (1 + (beta * t_actual));
        iter += 1;
    }
    return best_w;
}

double busquedaLocal(vector<vector<double>> &muestra, vector<char> &clase, vector<double> &w)
{
    const int maxIter = 1000;
    int cont = 0;
    double varianza = 0.3, alpha = 0.5;

    // Creo vector z y un generador de distribución normal
    vector<double> z(w.size());
    vector<double>::iterator z_it;
    normal_distribution<double> normal_dist(0.0, sqrt(varianza));
    double s = r();
    mt19937 other_eng(Seed);
    auto genNormalDist = [&normal_dist, &other_eng]()
    {
        return normal_dist(other_eng);
    };

    double fun_objetivo = 0;
    double max_fun = -99999.0;
    double w_aux;

    fun_objetivo = evalua(muestra, clase, w);
    cont++;
    max_fun = fun_objetivo;

    // Mientras no se superen las iteraciones máximas o los vecinos permitidos
    while (cont < maxIter)
    {
        generate(begin(z), end(z), genNormalDist);
        z_it = z.begin();

        for (vector<double>::iterator it = w.begin(); it != w.end(); it++)
        {
            // Guardamos w original
            w_aux = *it;

            // Mutación normal
            *it += *z_it;

            if (*it < 0)
                *it = 0;
            else if (*it > 1)
                *it = 1;

            fun_objetivo = evalua(muestra, clase, w);
            cont++;

            // Si hemos mejorado el umbral a mejorar cambia, vamos maximizando la función
            if (fun_objetivo > max_fun)
                max_fun = fun_objetivo;
            else // Si no hemos mejorado nos quedamos con la w anterior
                *it = w_aux;
            z_it++;
        }
    }
    return max_fun;
}

vector<double> algILS(vector<vector<double>> &muestra, vector<char> &clase)
{
    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    // Inizializo vector de pesos con el generador
    vector<double> w(muestra.begin()->size());
    vector<double> best_w(muestra.begin()->size());
    generate(begin(w), end(w), gen);

    int num_mut = 0.1 * muestra.begin()->size();
    double eval = 0, best_eval = 0;
    int iter = 0, index = 0;
    double varianza = 0.4;

    // Creo vector z y un generador de distribución normal
    vector<double> z(w.size());
    vector<double>::iterator z_it;
    normal_distribution<double> normal_dist(0.0, sqrt(varianza));
    double s = r();
    mt19937 other_eng(Seed);
    auto genNormalDist = [&normal_dist, &other_eng]()
    {
        return normal_dist(other_eng);
    };

    best_eval = evalua(muestra, clase, w);
    best_w = w;

    eval = busquedaLocal(muestra, clase, w);

    if (eval > best_eval)
    {
        best_eval = eval;
        best_w = w;
    }

    w = best_w;
    iter++;

    while (iter < 15)
    {
        generate(begin(z), end(z), genNormalDist);
        z_it = z.begin();
        for (int i = 0; i < num_mut; i++)
        {
            // comienzo mutaciones
            index = r() % w.size();
            w[index] += *z_it;
            z_it++;

            if (w[index] < 0.0)
                w[index] = 0.0;
            else if (w[index] > 1.0)
                w[index] = 1.0;
        }

        eval = busquedaLocal(muestra, clase, w);

        if (eval > best_eval)
        {
            best_eval = eval;
            best_w = w;
        }
        iter++;
        w = best_w;
    }
    return best_w;
}

vector<double> algILS_ES(vector<vector<double>> &muestra, vector<char> &clase)
{
    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    // Inizializo vector de pesos con el generador
    vector<double> w(muestra.begin()->size());
    vector<double> best_w(muestra.begin()->size());
    generate(begin(w), end(w), gen);

    int num_mut = 0.1 * muestra.begin()->size();
    double eval = 0, best_eval = 0;
    int iter = 0, index = 0;
    double varianza = 0.4;

    // Creo vector z y un generador de distribución normal
    vector<double> z(w.size());
    vector<double>::iterator z_it;
    normal_distribution<double> normal_dist(0.0, sqrt(varianza));
    double s = r();
    mt19937 other_eng(Seed);
    auto genNormalDist = [&normal_dist, &other_eng]()
    {
        return normal_dist(other_eng);
    };

    best_eval = evalua(muestra, clase, w);
    best_w = w;

    w = algEnfriamientoSimulado(muestra, clase);
    eval = evalua(muestra, clase, w);

    if (eval > best_eval)
    {
        best_eval = eval;
        best_w = w;
    }

    w = best_w;
    iter++;

    while (iter < 15)
    {
        generate(begin(z), end(z), genNormalDist);
        z_it = z.begin();
        for (int i = 0; i < num_mut; i++)
        {
            // comienzo mutaciones
            index = r() % w.size();
            w[index] += *z_it;
            z_it++;

            if (w[index] < 0.0)
                w[index] = 0.0;
            else if (w[index] > 1.0)
                w[index] = 1.0;
        }

        w = algEnfriamientoSimulado(muestra, clase);
        eval = evalua(muestra, clase, w);

        if (eval > best_eval)
        {
            best_eval = eval;
            best_w = w;
        }
        iter++;
        w = best_w;
    }
    return best_w;
}

vector<double> algBMB(vector<vector<double>> &muestra, vector<char> &clase)
{
    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    // Inizializo vector de pesos con el generador
    vector<double> w(muestra.begin()->size());
    vector<double> best_w(muestra.begin()->size());
    generate(begin(w), end(w), gen);

    int iter = 0;
    double best_eval = 0, eval = 0;

    best_eval = busquedaLocal(muestra, clase, w);
    best_w = w;
    iter++;

    while (iter < 15)
    {
        mt19937 eng(r());
        uniform_real_distribution<double> dist(0.0, 1.0);
        auto gen = [&dist, &eng]()
        {
            return dist(eng);
        };
        generate(begin(w), end(w), gen);
        eval = busquedaLocal(muestra, clase, w);

        if (eval > best_eval)
        {
            best_eval = eval;
            best_w = w;
        }
        iter++;
    }
    return best_w;
}

int main(int nargs, char *args[])
{
    char *arg[4];
    string option;
    string path;

    if (nargs <= 2)
    {
        cerr << "[ERROR] Wrong execution pattern" << endl;
        cerr << "[Ex.] ./main {seed} [1-3] " << endl;
        cerr << "[Pd:] 1=spectf-heart, 2=parkinsons, 3=ionosphere" << endl;
    }
    Seed = atof(args[1]);
    option = args[2];

    if (option == "1")
        path = "./bin/spectf-heart.arff";
    else if (option == "2")
        path = "./bin/parkinsons.arff";
    else if (option == "3")
        path = "./bin/ionosphere.arff";
    else
    {
        cerr << "[ERROR] Parámetro no reconocido..." << endl;
        cerr << "[Ex.] Tienes que definir que data-set: 1-spectf-heart, 2-parkinsons, 3-ionosphere..." << endl;
        cerr << "[Ex.] ./main {seed} [1-3] " << endl;
        exit(1);
    }

    readData(path);
    normalizeData(data_matrix);

    pair<vector<vector<vector<double>>>, vector<vector<char>>> part;
    part = createPartitions();

    srand(Seed);
    std::cout << "\nSemilla: " << setprecision(10) << Seed << endl;

    std::cout << "\n------------(ALGORITMO ENFRIAMIENTO SIMULADO)------------\n\n";
    execute(part, algEnfriamientoSimulado);
    std::cout << "\n\n------------(ALGORITMO BÚSQUEDA LOCAL REITERADA)------------\n\n";
    execute(part, algILS);
    std::cout << "\n\n------------(ALGORITMO HÍBRIDO ILS-ES)------------\n\n";
    execute(part, algILS_ES);
    std::cout << "\n\n------------(ALGORITMO BÚSQUEDA MULTIARRANQUE BÁSICA)------------\n\n";
    execute(part, algBMB);

    std::cout << endl
              << endl;
}
