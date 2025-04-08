import os
import numpy as np
import networkx as nx
from pyverilog.ast_code_generator import ASTCodeGenerator
from pyverilog.ast_parser import parse
from pyverilog.ast_parser import ast as verilog_ast
from gensim.models import Word2Vec
from data_processing.graph_builder import GraphBuilder

class RTLProcessor:
    def __init__(self, config):
        self.ast_parser = ASTCodeGenerator()
        self.graph = nx.DiGraph()
        self.config = config
        self.word2vec_model = None
        self.graph_builder = GraphBuilder()

    def parse_verilog(self, file_path):
        """Parse Verilog file and generate AST"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Verilog file not found: {file_path}")
            
        ast, _ = parse([file_path])
        
        # Use either the graph builder or build graph directly
        if self.config.get('use_graph_builder', False):
            return self.graph_builder.build_from_ast(ast)
        else:
            return self._build_graph(ast)

    def _build_graph(self, ast):
        """Convert AST to directed graph with enhanced features"""
        for node in ast.children():
            node_id = hash(node)
            self.graph.add_node(node_id, 
                              type=type(node).__name__,
                              lineno=node.lineno,
                              code=str(node),
                              fan_in=0,
                              fan_out=0)
            
            for child in node.children():
                if isinstance(child, verilog_ast.SensList):
                    continue
                child_id = hash(child)
                self.graph.add_edge(node_id, child_id)
                
                # Update fan-in/fan-out counts
                self.graph.nodes[node_id]['fan_out'] += 1
                self.graph.nodes[child_id]['fan_in'] += 1
        return self.graph

    def generate_embeddings(self):
        """Generate node embeddings using Word2Vec"""
        walks = self._perform_random_walks()
        self.word2vec_model = Word2Vec(
            walks,
            vector_size=self.config['embedding_size'],
            window=self.config['window_size'],
            min_count=1,
            workers=4
        )
        
        # Create a dictionary of node embeddings
        node_embeddings = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node]['type']
            if node_type in self.word2vec_model.wv:
                node_embeddings[node] = self.word2vec_model.wv[node_type]
            else:
                # Fallback for unseen node types
                node_embeddings[node] = np.zeros(self.config['embedding_size'])
                
        return node_embeddings

    def _perform_random_walks(self):
        """Generate random walks for graph nodes"""
        walks = []
        nodes = list(self.graph.nodes())
        
        num_walks = self.config.get('num_walks', 20)
        walk_length = self.config.get('walk_length', 40)
        
        for _ in range(num_walks):
            np.random.shuffle(nodes)
            if not nodes:  # Check if nodes list is empty
                continue
                
            walk = [nodes[0]]
            
            while len(walk) < walk_length:
                current = walk[-1]
                neighbors = list(self.graph.neighbors(current))
                
                if neighbors:
                    walk.append(np.random.choice(neighbors))
                else:
                    break
                    
            walks.append([str(self.graph.nodes[n]['type']) for n in walk])
            
        return walks
