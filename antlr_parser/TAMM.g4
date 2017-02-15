grammar TAMM;

options {
    language = Cpp;
}

/*@parser::includes {
    #include <iostream>
    #include <list>
    #include <sstream>
    #include <string>
    using namespace std;
}*/


// TOKENS 
    
// Reserved Keywords
RANGE   :   'range';
INDEX  :   'index';
ARRAY  :   'array';
EXPAND :   'expand';
VOLATILE : 'volatile';
ITERATION : 'iteration';

// Operators
PLUS    :   '+';
MINUS   :   '-';
TIMES   :   '*';

// Assignment operators
EQUALS   :   '=';
TIMESEQUAL : '*=';
PLUSEQUAL : '+=';
MINUSEQUAL : '-=';

// Delimeters
LPAREN  :   '(';
RPAREN  :   ')';
LBRACE  :   '{';
RBRACE  :   '}';
LBRACKET  :   '[';
RBRACKET  :   ']';
COMMA   :   ',';
COLON   :   ':';
SEMI:  ';';

// Identifier   
ID
    :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
    ;

// Integer Constant
ICONST
    :   '0'..'9'+
    ;

FRAC
    :   ('1'..'9')+ '/' ('1'..'9')+
    ;

// Foalting Point Constant
FCONST
    :   ('0'..'9')+ '.' ('0'..'9')* EXPONENT?
    |   '.' ('0'..'9')+ EXPONENT?
    |   ('0'..'9')+ EXPONENT
    ;

fragment EXPONENT
    :   ('e'|'E') ('+'|'-')? ('0'..'9')+
    ;
    
    

// translation-unit
translation_unit : compound_element_list EOF ;

compound_element_list: (compound_element)* ;

element_list: (element)* ; 

// compound-element
compound_element : identifier LBRACE element_list RBRACE ;

       

// element
element : 
         declaration 
         |
         statement ;


// declaration
declaration : range_declaration 
              |       
              index_declaration 
              |
              array_declaration 
              |
              expansion_declaration 
              |
              volatile_declaration 
              |
              iteration_declaration ;


// id-list
id_list_opt : 
             |
             id_list ;


id_list : identifier (COMMA identifier)*;
              
num_list : numerical_constant (COMMA numerical_constant)*;       


// identifier
identifier : ID ;
    

// numerical-constant
numerical_constant : ICONST 
                     |
                     FCONST
                     |
                     FRAC;
    
    
    
// range-declaration
range_declaration : RANGE id_list EQUALS numerical_constant SEMI ;


// index-declaration
index_declaration : INDEX id_list EQUALS identifier SEMI ;


// array-declaration
array_declaration : ARRAY array_structure_list (COLON identifier)? SEMI ;

// array-structure
// Old - array_structure : ID LPAREN LBRACKET id_list_opt RBRACKET LBRACKET id_list_opt RBRACKET (permut_symmetry)? RPAREN ;

array_structure : ID LBRACKET id_list_opt RBRACKET LBRACKET id_list_opt RBRACKET (permut_symmetry)?;

array_structure_list : array_structure (COMMA array_structure)* ;



// permutational-symmetry
permut_symmetry : COLON (symmetry_group)+ ;
                          
                            
symmetry_group : LPAREN num_list RPAREN ;

expansion_declaration : EXPAND id_list SEMI ;

// volatile-declaration
volatile_declaration : VOLATILE id_list SEMI ;

// iteration-declaration
iteration_declaration : ITERATION EQUALS numerical_constant SEMI ;


// statement
statement : assignment_statement ;


// assignment-statement
assignment_statement : (identifier COLON)? expression assignment_operator expression SEMI ;


// assignment_operator
assignment_operator : EQUALS
                           | TIMESEQUAL
                           | PLUSEQUAL
                           | MINUSEQUAL ;
                           

// unary-expression
unary_expression : primary_expression 
                   |
                   PLUS unary_expression 
                   |
                   MINUS unary_expression ;
    
    
    
// primary-expression    
primary_expression : numerical_constant 
                     |
                     array_reference 
                     |
                     LPAREN expression RPAREN ;


// array-reference
array_reference : ID (LBRACKET id_list_opt RBRACKET)? ;


// expression                           
plusORminus : PLUS | MINUS ;

// additive-expression
expression :  multiplicative_expression (plusORminus multiplicative_expression)* ;


// multiplicative-expression
multiplicative_expression : unary_expression (TIMES unary_expression)* ;
                            

Whitespace
    :   [ \t]+
        -> skip
    ;

Newline
    :   (   '\r' '\n'?
        |   '\n'
        )
        -> skip
    ;

BlockComment
    :   '/*' .*? '*/'
        -> skip
    ;

LineComment
    :   '//' ~[\r\n]*
        -> skip
    ;

