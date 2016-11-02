
#include "gft.h"
#include "graph.h"

//Calcula o centro de gravidade da fatia fornecida
//(media ponderada das posicoes pela intensidade).
int CenterOfGravity(gft::Image32::Image32 *img){
  int p,x,y,val;
  float sx,sy,simg;
  simg = sx = sy = 0.0;
  for(p = 0; p < img->n; p++){
    x = p%img->ncols;
    y = p/img->ncols;
    val = img->data[p];
    sx += val*x;
    sy += val*y;
    simg += val;
  }
    sx /= simg;
    sy /= simg;
    return (((int)sx) + ((int)sy)*img->ncols);
}


//Adiciona no vetor S de sementes todos os pixels
//dentro de um raio r ao redor do pixel p e pinta suas
//posicoes correspondentes na imagem de label com o valor
//dado em lb. Note que S[0] por convencao armazena o
//numero de sementes.
void AddSeedsInCircle(int S[], gft::Image32::Image32 *label,
		      int p, float r, int lb){
  gft::AdjRel::AdjRel *A;
  int i,q,u_x,u_y,v_x,v_y;
  A = gft::AdjRel::Circular(r);

  u_x = p%label->ncols;
  u_y = p/label->ncols;
  for(i=0; i<A->n; i++){
    v_x = u_x + A->dx[i];
    v_y = u_y + A->dy[i];
    if(gft::Image32::IsValidPixel(label,v_x,v_y)){
      q = v_x + label->ncols*v_y;
      S[0]++;
      S[S[0]] = q;
      label->data[q] = lb;
    }
  }
  gft::AdjRel::Destroy(&A);
}

//Adiciona no vetor S de sementes todos pixels na borda da imagem
//e pinta suas posicoes correspondentes na imagem de label.
//Note que S[0] por convencao armazena o numero de sementes.
void AddSeedsInImageBorder(int S[], gft::Image32::Image32 *label, int lb){
  int px,py,p;
  for(px = 0; px < label->ncols; px++){
    p = px;
    if(label->data[p] == NIL){
      label->data[p] = lb;
      S[0]++;
      S[S[0]] = p;
    }

    p = px + (label->nrows-1)*label->ncols;
    if(label->data[p] == NIL){
      label->data[p] = lb;
      S[0]++;
      S[S[0]] = p;
    }
  }
  for(py = 1; py < label->nrows-1; py++){
    p = py*label->ncols;
    if(label->data[p] == NIL){
      label->data[p] = lb;
      S[0]++;
      S[S[0]] = p;
    }

    p = label->ncols-1 + py*label->ncols;
    if(label->data[p] == NIL){
      label->data[p] = lb;
      S[0]++;
      S[S[0]] = p;
    }
  }
}


//dilata parte inferior para recuperar cerebelo.
void FixCerebellum(gft::Scene32::Scene32 *label){
  gft::Scene32::Scene32 *tmp;
  tmp = gft::Scene32::Clone(label);
  int p,sx,sy,sz,sn;
  int x,y,z;
  sn = sx = sy = sz = 0;
  for(p = 0; p < label->n; p++){
    if(label->data[p] > 0){
      sx += gft::Scene32::GetAddressX(label, p);
      sy += gft::Scene32::GetAddressY(label, p);
      sz += gft::Scene32::GetAddressZ(label, p);
      sn++;
    }
  }
  sx /= sn;
  sy /= sn;
  sz /= sn;
  //printf("(%d,%d,%d)\n",sx,sy,sz);
  for(z = 0; z < label->zsize; z++){
    for(y = 0; y <= sy; y++){
      for(x = sx; x < label->xsize; x++){
	if(label->array[z][y+1][x] > 0 ||
	   label->array[z][y+2][x] > 0 ||
	   label->array[z][y+3][x] > 0 ||
	   label->array[z][y+4][x] > 0 ||
	   label->array[z][y+5][x] > 0)
	  tmp->array[z][y][x] = 1;
      }
    }
  }
  for(p = 0; p < label->n; p++){
    if(tmp->data[p] > 0)
      label->data[p] = 1;
  }
  gft::Scene32::Destroy(&tmp);
}

void AddNeighbors(Graph *g, Graph::node_id *nodes, gft::Image32 *img, int i) {
    int x = i%img->ncols;
    int y = i/img->ncols;
    int k = 0, m, n;
    int neighbors[] = {{x+1, y}, {x, y+1}, {x-1, y}, {x, y-1}};
    for (; k < 4; k++) {
        m = neighbors[k][0];
        n = neighbors[k][1];
        if(gft::Image32::IsValidPixel(img,m,n)){
            nodes
            g->add_egde(node, )
        }
    }
}



int main(int argc, char **argv){
  gft::Scene32::Scene32 *scn, *scnlabel;
  gft::Image32::Image32 *img, *imglabel, *grad, *P_sum;
  gft::SparseGraph::SparseGraph *sg;
  char filename[512];
  int S[5000];
  int y,p,center;

  if(argc < 2){
    fprintf(stdout,"usage:\n");
    fprintf(stdout,"proj2 <scene>\n");
    exit(0);
  }

  scn = gft::Scene32::Read(argv[1]);
  scnlabel = gft::Scene32::Clone(scn);

  //VAMOS TESTAR PARA APENAS UMA FATIA
  //for(y = 0; y < scn->ysize; y++){
    //pega uma fatia axial do volume.
    y = 141;
    img = gft::Scene32::GetSliceY(scn, y);
    //cria uma imagem de rotulos inicialmente com valor NIL = -1.
    imglabel = gft::Image32::Clone(img);
    gft::Image32::Set(imglabel, NIL);

    //cria um grafo com pesos nao direcionados a partir da fatia.
    sg = gft::SparseGraph::ByAccAbsDiff(img, 1.5, 2.0);
    //converte o grafo em um grafos direcionado.
    gft::SparseGraph::Orient2Digraph(sg, img, 70);

    //coloca sementes internas ao redor do centro de gravidade do volume.
    S[0] = 0;
    center = CenterOfGravity(img);
    AddSeedsInCircle(S, imglabel, center, 35.0, 1);

    //calcula IFT com funcao aditiva a partir das sementes internas
    //para definir os caminhos geodesicos a partir de centro
    //para usar a restricao de forma geodesica em estrela.
    P_sum = gft::ift::pred_IFTSUM(sg, S, NULL, 0.2, 2);
    gft::SparseGraph::Orient2DigraphInner(sg, P_sum);

    //coloca sementes de fundo na borda da imagem.
    AddSeedsInImageBorder(S, imglabel, 0);

    //Calcula OIFT no grafo favorecendo cortes do escuro para claro.
    gft::ift::method_IFTW_FIFO_InnerCut(sg, S, imglabel);

    //guarda resultado da fatia em volume de rotulos.
    gft::Scene32::PutSliceY(scnlabel, imglabel, y);

    //libera memoria usada.
    gft::SparseGraph::Destroy(&sg);
    gft::Image32::Destroy(&imglabel);
    gft::Image32::Destroy(&img);
    gft::Image32::Destroy(&grad);
    gft::Image32::Destroy(&P_sum);
  //}

  //dilata parte inferior para recuperar cerebelo.
  FixCerebellum(scnlabel);

  //aplica um filtro moda no resultado final para suavizar.
  gft::Scene32::ModeFilterLabel(scnlabel, 2.0);

  //grava volume de rotulos.
  gft::Scene32::Write(scnlabel, (char *)"label.scn.bz2");

  //gravando resultados de todas fatias.
  //for(y = 0; y < scn->ysize; y++){
    //pega uma fatia axial do volume.
    img = gft::Scene32::GetSliceY(scn, y);
    imglabel = gft::Scene32::GetSliceY(scnlabel, y);

    for(p = 0; p < img->n; p++){
      if(imglabel->data[p] == 0)
	  img->data[p] = 0;
    }

//GRAPHCUT
    Graph::node_id nodes[img->n];
    Graph *g = new Graph();

    nodes[0]      = g->add_node();
    nodes[center] = g->add_node();
    g->set_tweights(nodes[0], 0, 99999);
    g->set_tweights(nodes[center], 99999, 0);

    for (int i = 1; i < img->n; i++) {
        if (i != center) {
            nodes[i] = g->add_node();
            g->add_edge(nodes[i])
        }
    }

///////////////////




    sprintf(filename, "seg_%03d.pgm", y);
    gft::Image32::Write(img, filename);

    //libera memoria usada.
    gft::Image32::Destroy(&img);
    gft::Image32::Destroy(&imglabel);
  //}

  //libera memoria.
  gft::Scene32::Destroy(&scnlabel);
  gft::Scene32::Destroy(&scn);

  return 0;
}
