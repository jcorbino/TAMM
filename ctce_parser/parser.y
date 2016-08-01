    %include {

    #include "parser.h"
    #include "absyn.h"
    #include "error.h"
    
    void yyerror(char *s);

    }

    %nonassoc EQUALS TIMESEQUAL PLUSEQUAL MINUSEQUAL .
    %left PLUS MINUS .
    %left TIMES .

    %extra_argument   { TranslationUnit* root }
    %token_destructor { free($$);}

    %syntax_error {
      yyerror("Syntax error");
      //exit(1);
    }


    %parse_failure {
      fprintf(stderr,"Giving up.  Parser is lost...\n");
    }

    // translation-unit
    translation_unit ::= compound_element_list(C) . {
       *root = make_TranslationUnit(C);
    }


    // compound-element-list
    compound_element_list(C) ::=  . {
      C = make_CompoundElemList(NULL,NULL);
    }

    compound_element_list(C) ::= compound_element_list(L) compound_element(E) . {
        addTail_CompoundElemList(E,L);
        C = L;
    }


    // compound-element
    compound_element(C) ::= identifier LBRACE element_list(E) RBRACE . {
      CompoundElem ce = make_CompoundElem(E);
      C = ce; 
    }


    element_list(E) ::= . { E = make_ElemList(NULL,NULL); }

    element_list(E) ::= element_list(L) element(S)  . {
       addTail_ElemList(S,L);
       E = L;
    }
                   

    // element

    element(E) ::= declaration(D)  . { E = make_Elem_DeclList(D); }
    element(E) ::= statement(S) . { E = make_Elem_Stmt(S); }


    // declaration
    declaration(D) ::= range_declaration(R)  . { D = R; }
    declaration(D) ::= index_declaration(I)  . { D = I; }
    declaration(D) ::= array_declaration(A)  . { D = A; }
    declaration(D) ::= expansion_declaration(E) .  { D = E; }
    declaration(D) ::= volatile_declaration(V)  .  { D = V; }
    declaration(D) ::= iteration_declaration(I) .  { D = I; }


    // id-list
    id_list_opt(I) ::=  . { I = NULL; } //make_IDList(NULL,NULL); }
    id_list_opt(I) ::= id_list(L) . { I = L; }
        
        
    id_list(I) ::= identifier(N) . { I = make_IDList(N,NULL); }
    id_list(I) ::= id_list(L) COMMA identifier(N) . { 
      addTail_IDList(N,L);
      I = L;
     }
             
                  
    num_list(N) ::= numerical_constant(C) . { 
      ExpList e = make_ExpList(C,NULL); 
      N = e;
    }

    num_list(N) ::= num_list(L) COMMA numerical_constant(C) .  { 
      addTail_ExpList(C,L);
      N = L;
     }

    // identifier
    identifier(I) ::= ID(N) .  {
      Identifier id = make_Identifier(tce_tokPos,N);
      id->lineno = tce_lineno;
      I = id;
    }
        

    // numerical-constant
    numerical_constant(N) ::= ICONST(I)  . {
      Exp e = make_NumConst(tce_tokPos,atoi(I));
      e->lineno = tce_lineno;
      N = e;
    }

    numerical_constant(N) ::= FCONST(F)  . {
      Exp e = make_NumConst(tce_tokPos,atof(F));
      e->lineno = tce_lineno;
      N = e;      
    }

    numerical_constant(N) ::= FRACCONST(F)  . {
      char* frac = F;
      char* str = strdup(frac);
      char* pch;
      char* nd[2];
      pch = strtok (str," /");
      int i =0;
      while (pch != NULL)
      {
        nd[i] = pch; i++;
        pch = strtok (NULL, " /");
        
      }
      float num = atof(nd[0]);
      float den = atof(nd[1]);
      //printf("%f/%f\n",num,den);
      Exp e = make_NumConst(tce_tokPos,num/den);
      e->lineno = tce_lineno;
      N = e;
    }   
        
    // range-declaration
    range_declaration(R) ::= RANGE id_list(I) EQUALS numerical_constant(N) SEMI . {
      Exp e = N;
      float val = -1;
      if(e->kind == is_NumConst) val = e->u.NumConst.value;
      //TODO: Error check for e's type

      IDList p = I;
      DeclList dlist = make_DeclList(NULL,NULL);
      DeclList dl = dlist;  
      while(p != NULL){
        dl->head = make_RangeDecl(tce_tokPos,(p->head)->name,val);
        dl->head->lineno = (p->head)->lineno;
        p = p->tail;
        if(p!=NULL) {
          dl->tail = make_DeclList(NULL,NULL); 
          dl = dl->tail;
        }
      }

      dl = NULL;
      R = dlist;
    }


    // index-declaration
    index_declaration(D) ::= INDEX id_list(I) EQUALS identifier(N) SEMI . {
      Identifier e = N;

      IDList p = I;
      DeclList dlist = make_DeclList(NULL,NULL);
      DeclList dl = dlist;  
      while(p != NULL){
        dl->head = make_IndexDecl(tce_tokPos,(p->head)->name,mkString(e->name));
        dl->head->lineno = (p->head)->lineno;
        p = p->tail;
        if(p!=NULL) {
          dl->tail = make_DeclList(NULL,NULL); 
          dl = dl->tail;
        }
      }
      dl = NULL;
      D = dlist;
    }


    // array-declaration
      array_declaration(A) ::= ARRAY array_structure_list(S) SEMI . { A = S; }
      array_structure_list(A) ::= array_structure(S) . { A = make_DeclList(S,NULL); }
      array_structure_list(A) ::= array_structure_list(L) COMMA array_structure(S) . { 
        addTail_DeclList(S,L); 
        A = L;
      }

    // array-structure
    //old -- array_structure ::= ID LPAREN LBRACKET id_list_opt RBRACKET LBRACKET id_list_opt RBRACKET permut_symmetry_opt RPAREN .
    array_structure(A) ::= ID(N) LBRACKET id_list_opt(U) RBRACKET LBRACKET id_list_opt(L) RBRACKET permut_symmetry_opt(P) . {
     string id = N;
     IDList p = U;
     int countU = count_IDList(U);
     
     p = U;
     int ic = 0;
     string* indicesU = malloc(countU * sizeof(string));

     while (p!=NULL){
            indicesU[ic] = (p->head)->name;
            p = p->tail;
            ic++;
     } 

     int countL = count_IDList(L);
     ic = 0;
     p = L;
     string* indicesL = malloc(countL * sizeof(string));

     while (p!=NULL){
        indicesL[ic] = (p->head)->name;
        p = p->tail;
        ic++;
     } 

     Decl dec = make_ArrayDecl(tce_tokPos,id,indicesU,indicesL); 
     dec->u.ArrayDecl.ulen = countU;
     dec->u.ArrayDecl.llen = countL;
     dec->u.ArrayDecl.irrep = P; 
     dec->lineno = tce_lineno;
     A = dec;
     dec = NULL;
    }

    // statement
    statement(S) ::= assignment_statement(A) . { 
      Stmt st = A; 
      st->u.AssignStmt.label = NULL; 
      S=st;
    }

    statement(S) ::= ID(I) COLON assignment_statement(A) . { 
      Stmt st = A; 
      st->u.AssignStmt.label = I;
      S=st;
    }

    // assignment-statement
      assignment_statement(A) ::= expression(L) assignment_operator(O) expression(R) SEMI . { 
        Stmt s;
        Exp lhs = L;
        //lhs->lineno = tce_lineno;
        Exp rhs = R;
        //rhs->lineno = tce_lineno;
        string oper = O;

        if(strcmp(oper,"=")==0) { 
          s = make_AssignStmt(tce_tokPos,lhs,rhs); 
        }
        else {
          Exp tlhs = NULL;
          //if (lhs->kind == is_NumConst) tlhs = make_NumConst(tce_tokPos,0);
          //if (lhs->kind == is_Parenth)  tlhs = make_Parenth(tce_tokPos,NULL);
          
          //if (lhs->kind == is_Addition) tlhs = make_Addition(tce_tokPos,NULL);
          //if (lhs->kind == is_Multiplication) tlhs = make_Multiplication(tce_tokPos,NULL);
          
          if (lhs->kind == is_ArrayRef) {
            tlhs = make_Array(tce_tokPos,"",NULL); //create a copy of lhs for flattening
            //memcpy (tlhs, lhs, sizeof (lhs));
            tlhs->u.Array.name = mkString(lhs->u.Array.name);
            tlhs->u.Array.indices = mkIndexList(lhs->u.Array.indices,lhs->u.Array.length);
            tlhs->u.Array.length = lhs->u.Array.length;
            tlhs->lineno = lhs->lineno;
            tlhs->coef = lhs->coef;
          }
          assert(tlhs!=NULL); 

          Exp trhs = make_Parenth(tce_tokPos,rhs);
          //trhs->lineno = tce_lineno;

          if(strcmp(oper,"+=")==0) { 
            Exp tadd = make_Addition(tce_tokPos,make_ExpList(tlhs,make_ExpList(trhs,NULL)));        
            tadd->lineno = tlhs->lineno;
            s = make_AssignStmt(tce_tokPos,tlhs,tadd); 
          }
          else if(strcmp(oper,"-=")==0) { 
            trhs->coef *= -1;
            Exp tadd = make_Addition(tce_tokPos,make_ExpList(tlhs,make_ExpList(trhs,NULL)));   
            tadd->lineno = tlhs->lineno;           
            s = make_AssignStmt(tce_tokPos,tlhs,tadd); 
          }
            
          else if(strcmp(oper,"*=")==0) { 
            Exp tmult = make_Multiplication(tce_tokPos,make_ExpList(tlhs,make_ExpList(trhs,NULL)));      
            tmult->lineno = tlhs->lineno;     
            s = make_AssignStmt(tce_tokPos,tlhs,tmult); 
          }
        }
        s->u.AssignStmt.astype = oper;
        A = s;
      }


    // assignment_operator
    assignment_operator(O) ::= EQUALS(E) . { O = E; }
    assignment_operator(O) ::= TIMESEQUAL(T) . { O = T; }
    assignment_operator(O) ::= PLUSEQUAL(P) .  { O = P; }
    assignment_operator(O) ::= MINUSEQUAL(M) .  { O = M; }
                               

    // array-reference
    array_reference(A) ::= ID(N)  . {
      string id = N;
      Exp e = make_Array(tce_tokPos, id, NULL);
      e->lineno = tce_lineno;
      A = e;
    }

    array_reference(A) ::= ID(N) LBRACKET id_list(I) RBRACKET . {
     string id = N;

     IDList p = I;
     int count = count_IDList(p);
     p = I;

     string *indices = malloc(count * sizeof(string));

     int ic = 0;

      while (p!=NULL){
          indices[ic] = (p->head)->name;
          p = p->tail;
          ic++;
      } 

     Exp exp = make_Array(tce_tokPos, id, indices);
     exp->u.Array.length = count;
     exp->lineno = tce_lineno;
     A = exp;
     exp = NULL;
    }


    plusORMinus(O) ::= PLUS(P) . { O = P; }
    plusORMinus(O) ::= MINUS(M) . { O = M; }

    // expression                           
    expression(E) ::= additive_expression(A) . { 
      Exp e = A; 
      //e->lineno = tce_lineno;
      E = e;
    }

    additive_expression(A) ::=  multiplicative_expression(M) . { A = M; }

    additive_expression(E) ::= additive_expression(A) plusORMinus(O) multiplicative_expression(M) . {
     Exp e1 = A;
     Exp e2 = M;
     string op = O;
     ExpList el = make_ExpList(NULL,NULL);

     int clno = tce_lineno;
     if (e1->kind == is_Addition) e1->lineno = clno;

     if(strcmp(op,"-")==0) e2->coef *= -1;
     addTail_ExpList(e1,el);
     addTail_ExpList(e2,el);
     /*if (e1->kind == is_Addition) {
        addTail_ExpList(e1,el);
     }
     if (e2->kind == is_Addition) {
      addTail_ExpList(e2,el);
     }*/

     Exp nadd = make_Addition(tce_tokPos,el); 
     nadd->lineno = clno;
     E = nadd;
    }
    

    multiplicative_expression(M) ::= unary_expression(U) . { M = U; }
                                
    multiplicative_expression(E) ::= multiplicative_expression(M) TIMES unary_expression(U) . {
      Exp e1 = M;
      Exp e2 = U;

      ExpList el = make_ExpList(NULL,NULL);
      float coef = 1;
      int clno = tce_lineno;
      if (e1->kind == is_Multiplication) {
        clno = e1->lineno;
        coef *= e1->coef;
     }
     addTail_ExpList(e1,el);
     if (e2->kind == is_Multiplication) {
      coef *= e2->coef;
     }
     addTail_ExpList(e2,el);

      Exp nmult = make_Multiplication(tce_tokPos,el); 
      nmult->coef = coef;
      nmult->lineno = clno;
      E = nmult;
    }

    // unary-expression
    unary_expression(U) ::= primary_expression(E)  . { U = E; }
    unary_expression(U) ::= PLUS unary_expression(E)  . { U = E; }
    unary_expression(U) ::= MINUS unary_expression(E) . { 
      Exp ue = E;
      ue->coef *= -1;
      U = ue; 
    }
        
    // primary-expression    
    primary_expression(E) ::= numerical_constant(N) . { E = N; }
    primary_expression(E) ::= array_reference(A)  . { E = A; }
    primary_expression(E) ::= LPAREN expression(P) RPAREN . { E = make_Parenth(tce_tokPos,P); }


    // TODO: permutational-symmetry
    permut_symmetry_opt(P) ::=  . { P=NULL; }
    permut_symmetry_opt(P) ::=  COLON symmetry_group_list . { P=NULL; }
    //hack for array decl specifying irrep string - array X[upper][lower]:irrep_f
    permut_symmetry_opt(P) ::=  COLON ID(I) . { P=I; } 

    symmetry_group_list ::= symmetry_group .
    symmetry_group_list ::= symmetry_group_list symmetry_group .
                                
    symmetry_group ::= LPAREN num_list RPAREN .

    expansion_declaration ::= EXPAND id_list SEMI .

    // volatile-declaration
    volatile_declaration ::= VOLATILE id_list SEMI .

    // iteration-declaration
    iteration_declaration ::= ITERATION EQUALS numerical_constant SEMI .

