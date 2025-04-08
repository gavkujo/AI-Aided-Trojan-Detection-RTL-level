import networkx as nx
from pyverilog.ast_parser import ast

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_module = None
        self.signal_map = {}  # Maps signal names to node IDs

    def build_from_ast(self, ast_node):
        """Construct graph from Verilog AST"""
        for node in ast_node.children():
            if isinstance(node, ast.ModuleDef):
                self._process_module(node)
                # Process module items
                for item in node.items:
                    if isinstance(item, ast.Decl):
                        self._process_declaration(item)
                    elif isinstance(item, ast.Always):
                        self._process_always_block(item)
                    elif isinstance(item, ast.Assign):
                        self._process_assignment(item)
                    elif isinstance(item, ast.Instance):
                        self._process_instance(item)
        return self.graph

    def _process_module(self, node):
        """Add module node to graph"""
        self.current_module = node.name
        module_id = f"module_{node.name}"
        self.graph.add_node(module_id, 
                          type='module', 
                          name=node.name,
                          lineno=node.lineno)
        
        # Process ports
        for port in node.portlist.ports:
            port_id = f"port_{port.name}"
            self.graph.add_node(port_id, 
                              type='port',
                              name=port.name,
                              direction=port.direction,
                              lineno=node.lineno)
            self.graph.add_edge(module_id, port_id, relationship='has_port')
            self.signal_map[port.name] = port_id

    def _process_declaration(self, node):
        """Process wire/reg declarations"""
        for var in node.list:
            var_name = var.name
            var_id = f"signal_{var_name}"
            var_type = type(var).__name__
            
            self.graph.add_node(var_id, 
                              type=var_type,
                              name=var_name,
                              width=self._get_width(var.width),
                              lineno=node.lineno)
            
            self.graph.add_edge(f"module_{self.current_module}", var_id, 
                              relationship='declares')
            
            self.signal_map[var_name] = var_id

    def _process_always_block(self, node):
        """Process always blocks"""
        always_id = f"always_{id(node)}"
        self.graph.add_node(always_id, 
                          type='always',
                          sensitivity=self._get_sensitivity_list(node.sens_list),
                          lineno=node.lineno)
        
        self.graph.add_edge(f"module_{self.current_module}", always_id, 
                          relationship='contains')
        
        # Process statements in the always block
        self._process_statement(node.statement, always_id)

    def _process_statement(self, statement, parent_id):
        """Process statements recursively"""
        if statement is None:
            return
            
        stmt_id = f"stmt_{id(statement)}"
        stmt_type = type(statement).__name__
        
        self.graph.add_node(stmt_id, 
                          type=stmt_type,
                          lineno=statement.lineno)
        
        self.graph.add_edge(parent_id, stmt_id, relationship='contains')
        
        # Handle different statement types
        if isinstance(statement, ast.Block):
            for substmt in statement.statements:
                self._process_statement(substmt, stmt_id)
                
        elif isinstance(statement, ast.IfStatement):
            # Add condition
            cond_id = f"cond_{id(statement.cond)}"
            self.graph.add_node(cond_id, 
                              type='condition',
                              lineno=statement.lineno)
            self.graph.add_edge(stmt_id, cond_id, relationship='condition')
            
            # Add signals used in condition
            self._add_expression_signals(statement.cond, cond_id)
            
            # Process true and false branches
            if statement.true_statement:
                self._process_statement(statement.true_statement, stmt_id)
            if statement.false_statement:
                self._process_statement(statement.false_statement, stmt_id)
                
        elif isinstance(statement, ast.NonblockingSubstitution) or isinstance(statement, ast.BlockingSubstitution):
            # Handle assignments
            lhs_id = f"lhs_{id(statement.left)}"
            rhs_id = f"rhs_{id(statement.right)}"
            
            self.graph.add_node(lhs_id, type='lvalue', lineno=statement.lineno)
            self.graph.add_node(rhs_id, type='rvalue', lineno=statement.lineno)
            
            self.graph.add_edge(stmt_id, lhs_id, relationship='target')
            self.graph.add_edge(stmt_id, rhs_id, relationship='source')
            
            # Add signals used in left and right sides
            self._add_expression_signals(statement.left, lhs_id)
            self._add_expression_signals(statement.right, rhs_id)

    def _process_assignment(self, node):
        """Process continuous assignments"""
        assign_id = f"assign_{id(node)}"
        self.graph.add_node(assign_id, 
                          type='assign',
                          lineno=node.lineno)
        
        self.graph.add_edge(f"module_{self.current_module}", assign_id, 
                          relationship='contains')
        
        # Add left and right sides
        lhs_id = f"lhs_{id(node.left)}"
        rhs_id = f"rhs_{id(node.right)}"
        
        self.graph.add_node(lhs_id, type='lvalue', lineno=node.lineno)
        self.graph.add_node(rhs_id, type='rvalue', lineno=node.lineno)
        
        self.graph.add_edge(assign_id, lhs_id, relationship='target')
        self.graph.add_edge(assign_id, rhs_id, relationship='source')
        
        # Add signals used in left and right sides
        self._add_expression_signals(node.left, lhs_id)
        self._add_expression_signals(node.right, rhs_id)

    def _process_instance(self, node):
        """Process module instantiations"""
        instance_id = f"instance_{node.name}"
        self.graph.add_node(instance_id, 
                          type='instance',
                          name=node.name,
                          module=node.module,
                          lineno=node.lineno)
        
        self.graph.add_edge(f"module_{self.current_module}", instance_id, 
                          relationship='instantiates')
        
        # Process port connections
        for port_conn in node.portlist:
            port_id = f"port_conn_{id(port_conn)}"
            self.graph.add_node(port_id, 
                              type='port_connection',
                              port=port_conn.portname,
                              lineno=node.lineno)
            
            self.graph.add_edge(instance_id, port_id, relationship='connects')
            
            # Add signals used in port connection
            if port_conn.argname:
                self._add_expression_signals(port_conn.argname, port_id)

    def _add_expression_signals(self, expr, parent_id):
        """Extract signals from expressions and add to graph"""
        if expr is None:
            return
            
        # Handle different expression types
        if isinstance(expr, ast.Identifier):
            signal_name = expr.name
            if signal_name in self.signal_map:
                signal_id = self.signal_map[signal_name]
                self.graph.add_edge(parent_id, signal_id, relationship='uses')
                
        elif isinstance(expr, ast.Operator):
            # Process operands
            for operand in expr.nextnodes:
                self._add_expression_signals(operand, parent_id)
                
        elif isinstance(expr, ast.Concat):
            # Process concatenation items
            for item in expr.list:
                self._add_expression_signals(item, parent_id)
                
        elif isinstance(expr, ast.Partselect):
            # Process array/vector access
            self._add_expression_signals(expr.var, parent_id)
            self._add_expression_signals(expr.msb, parent_id)
            self._add_expression_signals(expr.lsb, parent_id)

    def _get_sensitivity_list(self, sens_list):
        """Extract sensitivity list from always block"""
        if not sens_list:
            return []
            
        sensitivity = []
        for item in sens_list.list:
            if isinstance(item, ast.Sens):
                edge = item.type
                signal = item.sig.name if item.sig else None
                sensitivity.append((edge, signal))
                
        return sensitivity

    def _get_width(self, width):
        """Extract width information from variable declaration"""
        if width is None:
            return 1
            
        if isinstance(width, ast.Width):
            msb = self._get_constant_value(width.msb)
            lsb = self._get_constant_value(width.lsb)
            if msb is not None and lsb is not None:
                return msb - lsb + 1
                
        return None

    def _get_constant_value(self, node):
        """Extract constant value from AST node"""
        if isinstance(node, ast.IntConst):
            return int(node.value)
        return None
