// this will be inserted at the top top 
%code requires {
#include "config_utils.hpp"
typedef void * yyscan_t;
}


%define api.pure full
%lex-param {yyscan_t scanner}
%parse-param {yyscan_t scanner}{Configuration &config}

%{
#include "config_utils.hpp"
#include "config.parser.tab.h"
#include "config.lex.h"

/* int  yylex(void); */
void yyerror (yyscan_t scanner, Configuration &config, char const *msg);
/* void yyerror(char * msg); */
/* void config_assign_string( char const * field, char const * value ); */
/* void config_assign_int( char const * field, int value ); */
/* void config_assign_float( char const * field, double value ); */
/**/
/* #ifdef _WIN32 */
/* #pragma warning ( disable : 4102 ) */
/* #pragma warning ( disable : 4244 ) */
/* #endif */

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

void yyerror (yyscan_t scanner, Configuration &config, char const *msg) {
	fprintf(stderr, "--> %s\n", msg);
}
