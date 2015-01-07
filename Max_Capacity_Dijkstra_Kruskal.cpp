/* 
 @author- Nishant Gill 
*/

#include<iostream>
#include<list>
#include<vector>
#include<set>
#include<algorithm>
#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<climits>
#include<cstring>
using namespace std;

#define V 5000
#define HEAP_CAPACITY 5000000
#define SIZE 5000000
#define MAX_WEIGHT 5000
#define UNSEEN 0
#define FRINGE 1
#define INTREE 2
#define RANDOM 1000
	 
int status[V],d[V],dad[V];

class Edge{
	public:
	int src;
	int dst;
	int weight;
	Edge(){}
	Edge(int a, int b, int c){
		src=a;
		dst=b;
		weight=c;
	}
	void printEdge(){
		cout<<src<<","<<dst<<","<<weight<<" ; ";
	}
};


class Node{
	public:
	int value;
	list<Edge> adj;
	Node(int v){
		value=v;
	}
	void addEdge(int a, int b, int c){
		Edge e(a,b,c);
		adj.push_back(e);	
	}
	
	void printAdjacency(){
		list<Edge>::iterator it;
		for(it=adj.begin();it!=adj.end();it++)
			(*it).printEdge();
		cout<<endl;	
	}
};



class Graph{
	public:
	vector<Node> vertices;
	Graph(){
		for(int i=0;i<V;i++){
			Node n(i);
			vertices.push_back(n);		
		}
	}
	void printGraph(){
		list<int>::iterator it;
		cout<<"Printing Graph"<<endl;
		for(int i=0; i<V;i++){
			cout<<i<<" : ";
			vertices[i].printAdjacency();
		}
	}	
};

/*************************************** Max heap ***************************************/

int getMaximumHeap(int heap[]){
	return heap[0];
}

void swap(int &a, int &b){
	int temp=a;
	a=b;
	b=temp;
}

void insertMaxHeap(int *heap, int &size, int val){
	if(size > HEAP_CAPACITY)
		return;
	size++;	
	heap[size]=val;
/*
	if(size == 0)
	return;
	int child=size;
	int dad=(child-1)/2;
	while((child > 0) && (heap[child] > heap[dad])){ 
		swap(heap[child],heap[dad]);
		child=dad;
		dad=(child-1)/2;
	}		
*/		
}

void deleteMaxHeap(int *heap, int &size){
	if(size < 0)
		return;
	heap[0]=heap[size];
	size--;	
/*	if(size < 0) return;
	int dad=0;
	int child=(2*dad)+1;
	while((child <= size) && (heap[dad] < heap[child])){
		swap(heap[child],heap[dad]);
		dad=child;
		child=(2*dad)+1;
	}
*/		
}


void heapifyUtil(int *heap, int size, int i){ 
	int leftChild = 2*i+1 ;
	int rightChild = 2*i + 2;
	int max = i;
	if (leftChild <= size && d[heap[leftChild]] > d[heap[i]])
	    max = leftChild;
	if (rightChild <= size && d[heap[rightChild]] > d[heap[max]])            
	    max = rightChild;
	if (max != i){
		swap(heap[i], heap[max]);
	    heapifyUtil(heap,size,max);
	}
}    


void heapify(int *heap, int size){
    if(size==0)
    	return;
    for (int i = (size-1)/2; i >= 0; --i)
        heapifyUtil(heap,size,i);    
}



void testHeap(){
	int a[50],s=-1;
	insertMaxHeap(a,s,3);
	insertMaxHeap(a,s,11);
	insertMaxHeap(a,s,18);
	insertMaxHeap(a,s,8);
	insertMaxHeap(a,s,20);
	insertMaxHeap(a,s,35);
	heapify(a,s);
	for(int i=0;i<=s;i++)cout<<a[i]<<" ";
	cout<<endl;
	deleteMaxHeap(a,s);
	heapify(a,s);
	for(int i=0;i<=s;i++)cout<<a[i]<<" ";

}


/************************************************Dijkstra********************************/

bool compare(int a, int b){
	return d[a] < d[b];
}


int Dijkstra_without_heap(Graph *g, int s, int dest){
	int i;
	list<int> fringes;
	for (i=0;i<V;i++){
		status[i]=UNSEEN;
		dad[i]=-2;
	}	
	status[s]=INTREE;
	d[s]=INT_MAX;
	dad[s]=-1;	
	Node *source = &(g->vertices[s]);
	list<Edge>::iterator it;
	list<int>::iterator itf;
	
	for (it=source->adj.begin(); it != source->adj.end(); it++){
		int w=(*it).dst;
		status[w]=FRINGE;
		d[w] = (*it).weight;
		fringes.push_back(w);
		dad[w]=s;	
	}
	while(!fringes.empty()){
		itf=max_element(fringes.begin(),fringes.end(),compare); 
		int v= *itf;
		fringes.erase(itf); 
		status[v]=INTREE;
		Node vertex = g->vertices[v];
		for(it=vertex.adj.begin();it!=vertex.adj.end();it++){
			int w=(*it).dst;
			int weight_v_w=(*it).weight;
			if (status[w] == UNSEEN){
				status[w]=FRINGE;
				d[w]=min(d[v],weight_v_w);
				fringes.push_back(w); 
				dad[w]=v;
			}
			else if ((status[w] == FRINGE) && (d[w] < min(d[v],weight_v_w))){
				d[w]= min(d[v],weight_v_w);
				dad[w]=v;
			}
		}	
	}
	return d[dest];
}


struct heapComp {
    bool operator() (int a, int b){
	return d[a] > d[b];
	}
};

int Dijkstra_with_heap(Graph *g, int s, int dest){
	int i;
	int fringes[2000000];
	int size=-1;	
	for (i=0;i<V;i++){
		status[i]=UNSEEN;
		dad[i]=-2;
	}		
	status[s]=INTREE;
	d[s]=INT_MAX;
	dad[s]=-1;	
	Node *source = &(g->vertices[s]);
	list<Edge>::iterator it;

	for (it=source->adj.begin(); it != source->adj.end(); it++){
		int w=(*it).dst;
		status[w]=FRINGE;
		d[w] = (*it).weight;
		insertMaxHeap(fringes,size,w);
		dad[w]=s;	
	}
	while(size != -1){
		heapify(fringes,size);	
		int v= getMaximumHeap(fringes);
		deleteMaxHeap(fringes,size); 
		status[v]=INTREE;

		Node *vertex = &(g->vertices[v]);
		for(it=vertex->adj.begin();it!=vertex->adj.end();it++){
			int w=(*it).dst;
			int weight_v_w=(*it).weight;
			if (status[w] == UNSEEN){
				status[w]=FRINGE;
				d[w]=min(d[v],weight_v_w);
				insertMaxHeap(fringes,size,w);
				dad[w]=v;
			}
			else if ((status[w] == FRINGE) && (d[w] < min(d[v],weight_v_w))){
				d[w]= min(d[v],weight_v_w);
				dad[w]=v;
			}
		}
	}
		return d[dest];
}




/*************************************Kruskal******************************************/

class makeset{
	public:
	int parent;
	int rank;
	makeset(){}
};


int find(makeset sets[], int i){
    if (sets[i].parent != i)
        sets[i].parent = find(sets, sets[i].parent);
    return sets[i].parent;
}

void Union(makeset sets[], int x, int y){
    int xroot = find(sets, x);
    int yroot = find(sets, y);

    if (sets[xroot].rank < sets[yroot].rank)
        sets[xroot].parent = yroot;
    else if (sets[xroot].rank > sets[yroot].rank)
        sets[yroot].parent = xroot;

    else{
        sets[yroot].parent = xroot;
        sets[xroot].rank++;
    }
}
 
// Compare two edges according to their weights
int edgeComp(Edge a, Edge b){
    return a.weight > b.weight;
}



void DFSUtil(Graph *g, int v, bool visited[],int d[],int dad[]){
    visited[v] = true;
    list<Edge>::iterator i;
    for( i = g->vertices[v].adj.begin(); i != g->vertices[v].adj.end(); ++i){
    	int dst = (*i).dst;
    	int weight = (*i).weight;
        if(!visited[dst]){
        	dad[dst]=v;
        	d[dst]=min(d[v],weight);
            DFSUtil(g,dst,visited,d,dad);            
        } 
    }
}
 
int DFS(Graph *g, int src, int dst){
    bool visited[V];
    int d[V],dad[V];
    memset(d,0,sizeof(d));
    memset(visited,0,sizeof(visited));
	for (int i=0;i<V;i++){
		dad[i]=-2;
	}	
    d[src]=INT_MAX;
    dad[src]=-1;
    DFSUtil(g,src,visited,d,dad);
    return d[dst];
}



void swap(Edge *x,Edge *y)
{
    Edge temp=*x;
    *x=*y;
    *y=temp;
}

void heapify_krus(vector<Edge> *heap, int &size, int i){ 
		int left = 2*i+1 ;
		int right = 2*i + 2;
		int max = i;
		Edge eleft = (*heap)[left];
		Edge eright = (*heap)[right];
		Edge ei = (*heap)[i];
		if (left < size && eleft.weight < ei.weight)
		    max = left;
		Edge emax =(*heap)[max]; 
		if (right < size && eright.weight < emax.weight)           
		    max = right;
	 
		if (max != i){
		    swap(&(*heap)[i], &(*heap)[max]);
		    heapify_krus(heap,size,max);
		}
	}    

void BuildMaxHeap(vector<Edge> *heap, int size){
	if(size==0)
    	return;
    for (int i = (size-1)/2; i >= 0; --i)
        heapify_krus(heap,size,i);    
    }


void HeapSort(vector<Edge> *heap, int n){
	int heapsize=n-1;
    BuildMaxHeap(heap,n);

    for(int i=heapsize;i>0;i--){
        swap(&(*heap)[0],&(*heap)[i]);
        heapify_krus(heap,heapsize,0);
        heapsize--;
    }
}

int KruskalMaxCap(Edge * result,int e,int src, int dst){
	Graph *g = new Graph();
	for (int i = 0; i < e; ++i){
		int src=result[i].src;
		int dst=result[i].dst;
		int weight=result[i].weight;
		g->vertices[src].addEdge(src,dst,weight);
		g->vertices[dst].addEdge(dst,src,weight);	
	}
return DFS(g,src,dst);
}

int KruskalMST(vector<Edge> *edge,int src, int dst){
    Edge mstedges[V]; 
    int e = 0; 
    int i = 0;  
    vector<Edge> edges = *edge;
    sort(edges.begin(), edges.end(), edgeComp);
    makeset *sets = new makeset[V];

    for (int v = 0; v < V; ++v){
        sets[v].parent = v;
        sets[v].rank = 0;
    }
    while (e < V - 1){
        Edge new_edge = edges[i++];
        int x = find(sets, new_edge.src);
        int y = find(sets, new_edge.dst);
        if (x != y){
            mstedges[e++] = new_edge;
            Union(sets, x, y);
        }
    }
    return KruskalMaxCap(mstedges,e,src,dst);
}


void testKruskal(vector<Edge> *edges){
	//Graph *g;
	edges->push_back(Edge(0,1,13));
	edges->push_back(Edge(0,2,10));
	edges->push_back(Edge(0,3,8));
	edges->push_back(Edge(1,4,11));
	edges->push_back(Edge(1,5,18));
	edges->push_back(Edge(5,6,10));
	edges->push_back(Edge(6,7,15));
	edges->push_back(Edge(4,8,15));
	edges->push_back(Edge(2,5,5));
	edges->push_back(Edge(3,6,21));
	edges->push_back(Edge(4,7,25));
	edges->push_back(Edge(6,9,6));
	edges->push_back(Edge(0,9,17));	
	
	//KruskalMST(edges,0,9);	

}


void testDijkstra_without_heap(){
	Graph g;
	g.vertices[0].addEdge(0,1,13);
	g.vertices[0].addEdge(0,2,10);
	g.vertices[0].addEdge(0,3,8);
	g.vertices[1].addEdge(1,4,11);
	g.vertices[1].addEdge(1,5,18);
	g.vertices[5].addEdge(5,6,10);
	g.vertices[6].addEdge(6,7,15);
	g.vertices[4].addEdge(4,8,15);
	g.vertices[2].addEdge(2,5,5);
	g.vertices[3].addEdge(3,6,21);
	g.vertices[4].addEdge(4,7,25);
	g.vertices[6].addEdge(6,9,6);
	g.vertices[1].addEdge(1,9,17);	

	//Dijkstra_without_heap(g,0,5);

}

void testDijkstra_with_heap(){
	Graph *g= new Graph() ;
	g->vertices[0].addEdge(0,1,13);
	g->vertices[1].addEdge(1,0,13);
	g->vertices[0].addEdge(0,2,9);
	g->vertices[2].addEdge(2,0,9);
	g->vertices[0].addEdge(0,3,8);
	g->vertices[3].addEdge(3,0,8);
	g->vertices[1].addEdge(1,4,11);
	g->vertices[4].addEdge(4,1,11);
	g->vertices[1].addEdge(1,5,18);
	g->vertices[5].addEdge(5,1,18);
	g->vertices[5].addEdge(5,6,10);
	g->vertices[6].addEdge(6,5,10);
	g->vertices[6].addEdge(6,7,15);
	g->vertices[7].addEdge(7,6,15);
	g->vertices[4].addEdge(4,8,16);
	g->vertices[8].addEdge(8,4,16);
	g->vertices[2].addEdge(2,5,5);
	g->vertices[5].addEdge(5,2,5);
	g->vertices[3].addEdge(3,6,21);
	g->vertices[6].addEdge(6,3,21);
	g->vertices[4].addEdge(4,7,25);
	g->vertices[7].addEdge(7,4,25);
	g->vertices[6].addEdge(6,9,6);
	g->vertices[9].addEdge(9,6,6);
	g->vertices[0].addEdge(0,9,17);	
	g->vertices[9].addEdge(9,0,17);	

	//cout<<Dijkstra_with_heap(g,6,0);

}




Graph* generateGraph(int branching_factor, vector<Edge> *edges){
	Graph *g= new Graph() ;
	srand(time(NULL));
	int count[V];
	list<Edge>::iterator it;
	memset(count,0,sizeof(count));
	bool is_neighbor[V];
	int val,weight,new_node,random_count; 
	
	for(int i=0; i<V; i++){
		memset(is_neighbor,0,sizeof(is_neighbor));
		if(count[i]>0){
			for (it=g->vertices[i].adj.begin(); it != g->vertices[i].adj.end(); it++)
				is_neighbor[(*it).dst]=true;		
		}
		while(count[i] < branching_factor){
			random_count=0;
			while(random_count < RANDOM){
				new_node=rand()%V;
				random_count++;
				if((is_neighbor[new_node] == false) && (i != new_node) && (count[new_node] < branching_factor))
					break;
			}
			if(random_count == RANDOM)
			break;
			
			weight=(rand()%MAX_WEIGHT)+1;
			g->vertices[i].addEdge(i,new_node,weight);
			g->vertices[new_node].addEdge(new_node,i,weight);
			edges->push_back(Edge(i,new_node,weight));
			is_neighbor[new_node]=true;
			count[i]+=1;
			count[new_node]+=1;
		}
	}
	return g;
}


void testing(){
	cout<<"Testing the code"<<endl;

	clock_t t,t1;
	
	cout<<"Testing for graphs with degree 6"<<endl;
	for(int i=0;i<5;i++){
		vector<Edge> *edges = new vector<Edge>();		
		Graph *g = generateGraph(6,edges);
		cout<<endl<<"New graph generated."<<endl;
		int src,dst;
		for(int j=0;j<5;j++){ 
			src=rand()%V;;
			while((dst=rand()%V) == src);
			cout<<endl<<"src: "<<src<<" -> dst: "<<dst<<endl;
			t=clock();
			cout<<"Max capacity distance using Dijkstra without heap is: "<<Dijkstra_without_heap(g,src,dst)<<endl;
			t1= clock() - t;
			cout<<"Dijkstra without heap completed in "<<(((float)t1)/CLOCKS_PER_SEC)<<" seconds"<<endl;
			t=clock();
			cout<<"Max capacity distance using Dijkstra with heap is: "<<Dijkstra_with_heap(g,src,dst)<<endl;
			t1= clock() - t;
			cout<<"Dijkstra with Heap completed in "<<(((float)t1)/CLOCKS_PER_SEC)<<" seconds"<<endl;
			t=clock();
			cout<<"Max capacity distance using Kruskal is: "<<KruskalMST(edges,src,dst)<<endl;
			t1= clock() - t;
			cout<<"Kruskal completed in "<<(((float)t1)/CLOCKS_PER_SEC)<<" seconds"<<endl;
		}	
	
	}
	
	cout<<endl<<"********************************************************************"<<endl;

	cout<<"Testing for graphs with degree 1000"<<endl;
	for(int i=0;i<5;i++){
		vector<Edge> *edges = new vector<Edge>();		
		Graph *g = generateGraph(1000,edges);
		cout<<endl<<"New graph generated."<<endl;
		int src,dst;
		for(int j=0;j<5;j++){ 
			src=rand()%V;;
			while((dst=rand()%V) == src);
			cout<<endl<<"src: "<<src<<" -> dst: "<<dst<<endl;
			t=clock();
			cout<<"Max capacity distance using Dijkstra without heap is: "<<Dijkstra_without_heap(g,src,dst)<<endl;
			t1= clock() - t;
			cout<<"Dijkstra without heap completed in "<<(((float)t1)/CLOCKS_PER_SEC)<<" seconds"<<endl;
			t=clock();
			cout<<"Max capacity distance using Dijkstra with heap is: "<<Dijkstra_with_heap(g,src,dst)<<endl;
			t1= clock() - t;
			cout<<"Dijkstra with Heap completed in "<<(((float)t1)/CLOCKS_PER_SEC)<<" seconds"<<endl;
			t=clock();
			cout<<"Max capacity distance using Kruskal is: "<<KruskalMST(edges,src,dst)<<endl;
			t1= clock() - t;
			cout<<"Kruskal completed in "<<(((float)t1)/CLOCKS_PER_SEC)<<" seconds"<<endl;
		}	
	
	}
	
}


int main(){
	testing();
	return 0;
}



