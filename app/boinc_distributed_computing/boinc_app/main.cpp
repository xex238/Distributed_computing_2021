/***********************************************************************\
 *  Hello, BOINC World!                                 Version: 7.01
 *
 *  This is the Hello World program for BOINC.  It is the simplest application
 *  one can write which uses the BOINC API and writes some output to a
 *  file (called "out.txt").   See the sample workunit and result templates
 *  to see how this file is mapped to a real file and how it is uploaded.
 *
 *  Note: if you want to run this program "standalone" then the file "out.txt"
 *  must already exist!
 *
 *  For more information see the release notes at
 *      http://www.spy-hill.com/help/boinc/hello.html
 *
 *  Eric Myers <myers@spy-hill.net> - 16 June 2004 (Unix)/6 July 2004 (Windows)
 *  @(#)  $Revision: 1.24 $ - $Date: 2012/03/26 02:50:31 $
\************************************************************************/

/*
#pragma comment(lib, "libboinc.a")
#pragma comment(lib, "libboinc_api.a")
#pragma comment(lib, "libboinc_crypt.a")
#pragma comment(lib, "libboinc_opencl.a")
#pragma comment(lib, "libcrypto.a")
#pragma comment(lib, "libcurl.a")
#pragma comment(lib, "libssl.a")
*/

#define _CRT_SECURE_NO_WARNINGS 1
# include <cstddef>		// now required for NULL, etc.


# ifdef _WIN32                //  Stuff we only need on Windows: 
# include "boinc_win.h"
# include "util.h"            // parse_command_line(), boinc_sleep()
#endif


/* BOINC API */

# include "boinc_api.h"
# include "diagnostics.h"     // boinc_init_diagnostics()
# include "filesys.h"         // boinc_fopen(), etc...
# include "str_util.h"		 // for parse_command_line()


// version.h is only used to get BOINC version numbers  
# include "version.h"           


/* Begin: */

int main(int argc, char** argv)
{
    int rc;                       // return code from various functions
    char resolved_name[512];      // physical file name for out.txt
    FILE* f;                      // file pointer for out.txt

    /*
     *  Before initializing BOINC itself, intialize diagnostics, so as
     *  to get stderr output to the file stderr.txt, and thence back home.
     */

    boinc_init_diagnostics(BOINC_DIAG_REDIRECTSTDERR |
        BOINC_DIAG_MEMORYLEAKCHECKENABLED |
        BOINC_DIAG_DUMPCALLSTACKENABLED |
        BOINC_DIAG_TRACETOSTDERR);

    /* Output written to stderr will be returned with the Result (task) */

    fprintf(stderr, "Hello, stderr!\n");


    /* BOINC apps that do not use graphics just call boinc_init() */

    rc = boinc_init();
    if (rc)
    {
        fprintf(stderr, "APP: boinc_init() failed. rc=%d\n", rc);
        fflush(0);
        exit(rc);
    }

    /*
     * Input and output files need to be "resolved" from their logical name
     * for the application to the actual path on the client's disk
     */
    rc = boinc_resolve_filename("out.txt", resolved_name, sizeof(resolved_name));
    if (rc)
    {
        fprintf(stderr, "APP: cannot resolve output file name. RC=%d\n", rc);
        boinc_finish(rc);    /* back to BOINC core */
    }

    /*
     *  Open files with boinc_fopen() not just fopen()
     *  (Output files should usually be opened in "append" mode, in case
     *  this is actually a restart (which will not be the case here)).
     */
    f = boinc_fopen(resolved_name, "a");

    fprintf(f, "Hello, BOINC World!\n");


    /* Now run up a wee bit of credit.   This is the "worker" loop */

    {
        int j, num, N;
        N = 123456789;
        fprintf(f, "Starting some computation...\n");
        for (j = 0; j < N; j++)
        {
            num = rand() + rand();     // just do something to spin the wheels
        }
        fprintf(f, "Computation completed.\n");
    }

    /* All BOINC applications must exit via boinc_finish(rc), not merely exit() */

    fclose(f);
    fprintf(stderr, "goodbye!\n");
    boinc_finish(0);       /* does not return */
}


/*
 *  Dummy graphics API entry points.  This app does not do graphics,
 *  but it still must provide these empty callbacks (for BOINC 5).
 */
#if BOINC_MAJOR_VERSION < 6
void app_graphics_init() {}
void app_graphics_resize(int width, int height) {}
void app_graphics_render(int xs, int ys, double time_of_day) {}
void app_graphics_reread_prefs() {}
void boinc_app_mouse_move(int x, int y, int left, int middle, int right) {}
void boinc_app_mouse_button(int x, int y, int which, int is_down) {}
void boinc_app_key_press(int, int) {}
void boinc_app_key_release(int, int) {}
#endif



# ifdef _WIN32 

/*******************************************************
 * Windows: Unix applications begin with main() while Windows applications
 * begin with WinMain, so this just makes WinMain() process the command line
 * and then invoke main()
 */

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR Args, int WinMode)
{
    LPSTR command_line;
    char* argv[100];
    int argc;

    command_line = (LPSTR)GetCommandLine();
    argc = parse_command_line(command_line, argv);
    return main(0, argv);
}

#endif
//EOF//