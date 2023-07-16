%{
#include "config_utils.hpp"
#include "config.parser.tab.h"
%}

%option reentrant
%option bison-bridge
%option bison-locations
%option noyywrap
%option 8bit
%option yylineno

Digit [0-9]
Exponent [eE][+-]?{Digit}+
DblConst ({Digit}*\.)?{Digit}+{Exponent}?
StrConst [A-Za-z_\-/\.][A-Za-z0-9_\-/\.\+(\{\,)\}]*


%%

  /* Ignore comments and all spaces */

\/\/[^\n]* ;
[ \t\r]*   ;

  /* Ignore new lines */

\n         { /*lineno++ handled by %option yylineno; */ }

  /* Commands */

\{[A-Za-z0-9_\-\.(\{\,)\}]+(\,[A-Za-z0-9_\-\.(\{\,)\}]+)*\} { yylval->name = strdup( yytext ); return STR; }

-?[0-9]+     { yylval->num = atoi( yytext ); return NUM; }

-?[0-9]*\.[0-9]+ { yylval->fnum = atof( yytext ); return FNUM; }

-?{DblConst}  { yylval->fnum = atof( yytext ); return FNUM;}

{StrConst} { yylval->name = strdup( yytext ); return STR; }

  /* Output all unrecognized ? */

.          { return yytext[0]; }

%%
