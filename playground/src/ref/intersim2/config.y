// this will be inserted at the very top, so that 
%code requires {
// so we can use the Configuration class in %parse-param
#include "config_utils.hpp"

// this avoids a cyclic include of config.lex.h and config.parser.tab.h
// because of this typedev we can use yyscan_t in the %parse-param and %lex-param
typedef void * yyscan_t;
}


%define api.pure full
%locations
%lex-param {yyscan_t scanner}
%parse-param {yyscan_t scanner}{Configuration &config}

%{
#include "config_utils.hpp"
#include "config.parser.tab.h"
#include "config.lex.h"

void yyerror(YYLTYPE* loc, yyscan_t scanner, Configuration &config, char const *msg);
%}

%union {
  char   *name;
  int    num;
  double fnum;
}

%token <name> STR
%token <num>  NUM
%token <fnum> FNUM

%%

commands : commands command
         | command
;

command : STR '=' STR ';'   { config.Assign( $1, $3 ); free( $1 ); free( $3 ); }
        | STR '=' NUM ';'   { config.Assign( $1, $3 ); free( $1 ); }
        | STR '=' FNUM ';'  { config.Assign( $1, $3 ); free( $1 ); }
;

%%

void yyerror(YYLTYPE* loc, yyscan_t scanner, Configuration &config, char const *msg) {
  int lineno = yyget_lineno(scanner);
  config.ParseError(msg, lineno);
}
