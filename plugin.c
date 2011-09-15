#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <gtk/gtk.h>
#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>
#include <libgimpwidgets/gimpwidgets.h>

#include <fftw3.h>

/** Defines ******************************************************************/

#define PLUG_IN_NAME "plug_in_addam_test"
#define PLUG_IN_BINARY "gimp-test"
#define PLUG_IN_AUTHOR "Addam Dominec"
#define PLUG_IN_VERSION "Aug 2011, 0.4.1"

#define GRAPH_WIDTH 200
#define GRAPH_HEIGHT 200
#define USER_POINT_COUNT 15

typedef struct _Curve {
	GdkPoint points[GRAPH_WIDTH];
	GdkPoint user_points[USER_POINT_COUNT];
	guchar count;
} Curve;

typedef struct PluginData {
  GimpDrawable      *drawable;
  GimpPixelRgn       region;

	gint              img_width, img_height, img_offset_x, img_offset_y;
	gint              img_bpp;
	fftw_complex    **image_freq; // array of pointers to image for each channel, frequency domain
	double          **image;  // same as above, image domain
	guchar           *img_pixels; // array used for acquiring data from the drawable
	fftw_plan         plan;

	Curve             curve_user, curve_preview, curve_fft;
	GtkWidget        *graph;
	GdkPixmap        *graph_pixmap;
} PluginData;

float curve_get_value(float x, Curve *c); // interpolate the curve at an arbitrary point in range [0; 1]

/** Plugin interface *********************************************************/

static void query(void);
static void run (const gchar      *name,
                 gint              nparams,
                 const GimpParam  *param,
                 gint             *nreturn_vals,
                 GimpParam       **return_vals);

GimpPlugInInfo PLUG_IN_INFO = {
  NULL,  /* init_proc  */
  NULL,  /* quit_proc  */
  query, /* query_proc */
  run    /* run_proc   */
};

void fft_prepare(PluginData *pd)
{
	gint         w = pd->img_width, h = pd->img_height;
	gint         img_bpp = pd->img_bpp, cur_bpp;
	int          x, y;
	double     **image = pd->image;
	guchar      *img_pixels;
	double       norm;
	image = pd->image = (double**) malloc(sizeof(double*) * img_bpp);
	pd->image_freq = (fftw_complex**) malloc(sizeof(fftw_complex*) * img_bpp);
  img_pixels = pd->img_pixels = g_new (guchar, w * h * img_bpp);
  //allocate an array for each channel
  for (cur_bpp=0; cur_bpp<img_bpp; cur_bpp ++){
	  image[cur_bpp] = (double*) fftw_malloc(sizeof(double) * w * h);
		pd->image_freq[cur_bpp] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (w/2+1) * h);
	}
	// forward plan
	fftw_plan plan = fftw_plan_dft_r2c_2d(pd->img_height, pd->img_width, *image, *pd->image_freq, FFTW_ESTIMATE);
	// inverse plan (to be reused)
	pd->plan = fftw_plan_dft_c2r_2d(pd->img_height, pd->img_width, *pd->image_freq, *image, FFTW_ESTIMATE);

	// execute forward FFT once
	gimp_pixel_rgn_init (&pd->region, pd->drawable, pd->img_offset_x, pd->img_offset_y, w, h, FALSE, FALSE);
	gimp_pixel_rgn_get_rect(&pd->region, img_pixels, pd->img_offset_x, pd->img_offset_y, w, h);
	gimp_pixel_rgn_init (&pd->region, pd->drawable, pd->img_offset_x, pd->img_offset_y, w, h, TRUE, TRUE);
	
	norm = 1.0/(w*h);
	for(cur_bpp=0; cur_bpp<img_bpp; cur_bpp++)
	{
		// convert one colour channel to double[]
		for(x=0; x < w; x ++)
		{
			for(y=0; y < h; y ++)
			{
				 image[cur_bpp][y*w + x] = norm * (double) img_pixels[(y*w + x)*img_bpp + cur_bpp];
			}
		}
		// transform the channel
		fftw_execute_dft_r2c(plan, image[cur_bpp], pd->image_freq[cur_bpp]);
	}
	fftw_destroy_plan(plan);
}

void fft_apply(PluginData *pd)
{
	int w = pd->img_width, h = pd->img_height,
		pw = w/2+1; // physical width
	fftw_complex *multiplied = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (w/2+1) * h);
	double diagonal = sqrt(h*h + w*w)/2.0;
	for(int cur_bpp=0; cur_bpp < pd->img_bpp; cur_bpp++)
	{
		// apply convolution
		for (int i=0; i < pw*h; i++){
			double coef;
			if (i==0) {//skip DC value FIXME: use a checkbox
				coef = 1.0;
			}
			else {
				int x, y;
				x = i % pw;
				y = i / pw;
				if (y>h/2){
					y = y-h;
				}
				double dist = sqrt(x*x + y*y);
				float point = 1.0 - (((diagonal/dist)-1.0) / (diagonal-1.0));
				if (point < 0 || point > 1)
					printf("%f -> %f!\n", dist, point);
				coef = curve_get_value(point, &pd->curve_user);
			}
			multiplied[i][0] = pd->image_freq[cur_bpp][i][0] * coef;
			multiplied[i][1] = pd->image_freq[cur_bpp][i][1] * coef;
		}
		// apply inverse FFT
		fftw_execute_dft_c2r(pd->plan, multiplied, pd->image[cur_bpp]);
		// pack results for GIMP
		for(int x=0; x < w; x ++)
		{
			for(int y=0; y < h; y ++)
			{
				double v = pd->image[cur_bpp][y*w + x];
				pd->img_pixels[(y*w + x) * pd->img_bpp+cur_bpp] = (v>255)?255:((v<0)?0:v);
			}
		}
	}
	fftw_free(multiplied);
}
void fft_destroy(PluginData *pd)
{
	fftw_destroy_plan(pd->plan);
	for (int i=0; i<pd->img_bpp; i++){
		fftw_free(pd->image[i]);
		fftw_free(pd->image_freq[i]);
	}
	free(pd->image);
	free(pd->image_freq);
}

void graph_update(PluginData *pd)
{
  GtkStyle *graph_style = gtk_widget_get_style (pd->graph);
	/*  Clear the pixmap  */
	gdk_draw_rectangle (pd->graph_pixmap, graph_style->bg_gc[GTK_STATE_NORMAL],
											TRUE, 0, 0, GRAPH_WIDTH, GRAPH_HEIGHT);
	
	/*  Draw the grid lines  */
	for (int i = 0; i < 5; i++)
	{
		gdk_draw_line (pd->graph_pixmap, graph_style->dark_gc[GTK_STATE_NORMAL],
									 0, i * (GRAPH_HEIGHT / 4),
									 GRAPH_WIDTH, (i*GRAPH_HEIGHT) / 4);
		gdk_draw_line (pd->graph_pixmap, graph_style->dark_gc[GTK_STATE_NORMAL],
									 (i*GRAPH_WIDTH) / 4, 0,
									 (i*GRAPH_WIDTH) / 4, GRAPH_HEIGHT);
	}
	gdk_draw_line (pd->graph_pixmap, graph_style->dark_gc[GTK_STATE_NORMAL],
								 0, GRAPH_HEIGHT - 1,
								 GRAPH_WIDTH, GRAPH_HEIGHT - 1);
	gdk_draw_line (pd->graph_pixmap, graph_style->dark_gc[GTK_STATE_NORMAL],
								 GRAPH_WIDTH - 1, 0,
								 GRAPH_WIDTH - 1, GRAPH_HEIGHT);
	gdk_draw_points (pd->graph_pixmap, graph_style->black_gc, pd->curve_user.points, GRAPH_WIDTH);
	gdk_draw_drawable (pd->graph->window, graph_style->black_gc, pd->graph_pixmap,
										 0, 0, 0, 0, GRAPH_WIDTH, GRAPH_HEIGHT);
}

void curve_init(Curve *c){
	c->count = 0;
	for (int i=0; i<GRAPH_WIDTH; i++){
		c->points[i].x = i;
		c->points[i].y = GRAPH_HEIGHT/2;
	}
}
// value -> graph
int value_to_graph(float val){
	if (val<1)
		return GRAPH_HEIGHT*(1.0-val/2);
	else
		return GRAPH_HEIGHT/(2*val);
}
// graph -> value
float graph_to_value(int y){
	if (y > GRAPH_HEIGHT/2)
		return ((float)GRAPH_HEIGHT - y) / (GRAPH_HEIGHT/2.0);
	else if (y>0)
		return (GRAPH_HEIGHT/2.0) / (float) y;
	else
		return GRAPH_HEIGHT/2.0;
}
// interpolate the curve at a given point between two points with indices i1, i2 (unnormalized)
float curve_interpolate(float x, int i1, int i2, Curve *c){
	// linear interpolation
	return ((x - c->user_points[i1].x) * graph_to_value(c->user_points[i2].y) + (c->user_points[i2].x - x) * graph_to_value(c->user_points[i1].y))
			/ (float) (c->user_points[i2].x - c->user_points[i1].x);
}
//get index to the first larger element
int bisect (int item, GdkPoint *array, int count){
	int left = 0, right = count;
	while (left != right){
		int test = (left+right)/2;
		if (array[test].x > item)
			right = test;
		else
			left = test+1;
	}
	return left;
}
// interpolate the curve at an arbitrary point in range [0; 1]
float curve_get_value(float x, Curve *c){
	if (c->count == 0)
		return 1.0;//No curve - constant 1
	x = x * GRAPH_WIDTH;
	int index = bisect(x, c->user_points, c->count);
	// extrapolation (constant... why not)
	if (index == c->count)
		return graph_to_value(c->user_points[index-1].y);
	else if (index == 0)
		return graph_to_value(c->user_points[index].y);
	else
		return curve_interpolate(x, index-1, index, c);
}
// insert a point into the curve
void curve_add_point(int x, int y, Curve *c){
	// get neighbours' positions (if any)
	GdkPoint point = {x, y};
	int i, index = bisect(x, c->user_points, c->count);
	for (i=c->count; i>index; i--){
		c->user_points[i] = c->user_points[i-1];
	}
	c->user_points[index] = point;
	c->count ++;
	// interpolate
	if (index > 0 && index + 1 < c->count)
		printf("Redraw between %i, %i, count:%i\n", c->user_points[index-1].x, c->user_points[index+1].x, c->count);
	if (index > 0)
		for (int i=c->user_points[index-1].x; i<x; i++){
			c->points[i].y = value_to_graph(curve_interpolate(i, index-1, index, c));
		}
	else
		for (int i=0; i<x; i++){
			c->points[i].y = y;
		}
	if (index+1 < c->count)
		for (int i=c->user_points[index+1].x; i>=x; i--){
			c->points[i].y = value_to_graph(curve_interpolate(i, index, index+1, c));
		}
	else
		for (int i=GRAPH_WIDTH-1; i>=x; i--){
			c->points[i].y = y;
		}
	// DEBUG
	/*for (int i=0; i<GRAPH_WIDTH; i++){
		printf("%.3i:%.2f, ", i, curve_get_value(((float)i)/GRAPH_WIDTH, c));
	}
	printf("\n");*/
}

static gint graph_events (GtkWidget *widget, GdkEvent *event, PluginData   *pd)
{
  static GdkCursorType cursor_type = GDK_TOP_LEFT_ARROW;
  GdkCursorType new_type;
  GdkEventButton *bevent;
  GdkEventMotion *mevent;
  int tx, ty;

  new_type      = GDK_X_CURSOR;

  /*  get the pointer position  */
  gdk_window_get_pointer (pd->graph->window, &tx, &ty, NULL);

	switch (event->type){
		case GDK_EXPOSE:
			printf("!\n");
			if (pd->graph_pixmap == NULL)
				pd->graph_pixmap = gdk_pixmap_new (pd->graph->window, GRAPH_WIDTH, GRAPH_HEIGHT, -1);
			graph_update (pd);
			break;
		
		case GDK_BUTTON_PRESS:
			bevent = (GdkEventButton *) event;
			new_type = GDK_TCROSS;
			
			graph_update (pd);
			break;
		
		case GDK_BUTTON_RELEASE:
			new_type = GDK_FLEUR;
			printf("clicked (%i, %i) -> value %f\n", tx, ty, graph_to_value(ty));
			curve_add_point(tx, ty, &pd->curve_user);
			graph_update (pd);
			break;
		
		case GDK_MOTION_NOTIFY:
			mevent = (GdkEventMotion *) event;
			
			if (mevent->is_hint)
			{
				mevent->x = tx;
				mevent->y = ty;
			}
			
			if (mevent->state & GDK_BUTTON1_MASK)
				new_type = GDK_TCROSS;
			else
				new_type = GDK_PENCIL;
			
			break;
		default:
			break;
	}
	if (new_type != cursor_type)
	{
		cursor_type = new_type;
		//change_win_cursor (pd->graph->window, cursor_type);
	}
	
	return FALSE;
}

gint preview_clicked(GtkWidget *widget, PluginData   *pd)
{
	fft_apply(pd);
	
	// show the result
	gimp_pixel_rgn_set_rect(&pd->region, pd->img_pixels, pd->img_offset_x, pd->img_offset_y, pd->img_width, pd->img_height);
	gimp_drawable_flush(pd->drawable);
	gimp_drawable_merge_shadow(pd->drawable->drawable_id, TRUE);
	gimp_drawable_update (pd->drawable->drawable_id, pd->img_offset_x, pd->img_offset_y, pd->img_width, pd->img_height);
	gimp_displays_flush();
}

gboolean dialog(PluginData *pd)
{
	gimp_ui_init (PLUG_IN_BINARY, FALSE);
	GtkWidget *dialog, *main_hbox, *preview, *graph, *vbox, *preview_button;
	dialog = gimp_dialog_new ("Frequency Curves", PLUG_IN_BINARY,
														NULL, (GtkDialogFlags)0,
														gimp_standard_help_func, PLUG_IN_NAME,
														GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
														GTK_STOCK_OK,     GTK_RESPONSE_OK,
														NULL);
	
	main_hbox = gtk_hbox_new (FALSE, 12);
	gtk_container_set_border_width (GTK_CONTAINER (main_hbox), 12);
	gtk_container_add (GTK_CONTAINER(GTK_DIALOG(dialog)->vbox), main_hbox);
	
	curve_init(&pd->curve_user);
	
	graph = pd->graph = gtk_drawing_area_new();
	pd->graph_pixmap = NULL;
	gtk_widget_set_size_request (graph, GRAPH_WIDTH, GRAPH_HEIGHT);
	gtk_widget_set_events (graph, GDK_EXPOSURE_MASK | GDK_POINTER_MOTION_MASK | 
	                              GDK_POINTER_MOTION_HINT_MASK | GDK_ENTER_NOTIFY_MASK | 
	                              GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK | 
	                              GDK_BUTTON1_MOTION_MASK);
	gtk_container_add (GTK_CONTAINER (main_hbox), graph);
	gtk_widget_show (graph);
	g_signal_connect (graph, "event", G_CALLBACK (graph_events), pd);
	
	vbox = gtk_vbox_new (FALSE, 12);
	gtk_container_set_border_width (GTK_CONTAINER (vbox), 12);
	gtk_container_add (GTK_CONTAINER(main_hbox), vbox);
	gtk_widget_show(vbox);
	
  preview_button = gtk_button_new_with_mnemonic ("HD _Preview");
  gtk_box_pack_start (GTK_BOX (vbox), preview_button, FALSE, FALSE, 0);
  gtk_widget_show (preview_button);
	g_signal_connect (preview_button, "clicked", G_CALLBACK (preview_clicked), pd);
	
	gtk_widget_show(main_hbox);
	printf(".\n");
	gtk_widget_show(dialog);
	fft_prepare(pd);
	printf("Ready.\n");
	gboolean run = (gimp_dialog_run (GIMP_DIALOG (dialog)) == GTK_RESPONSE_OK);
	printf(".\n");
	fft_destroy(pd);
	gtk_widget_destroy (dialog);
	return run;
}

/** Main GIMP functions ******************************************************/

MAIN()

void query(void)
{
  /* Definition of parameters */
  static GimpParamDef args[] = {
    { GIMP_PDB_INT32, (gchar *)"run_mode", (gchar *)"Interactive, non-interactive" },
    { GIMP_PDB_IMAGE, (gchar *)"image", (gchar *)"Input image (unused)" },
    { GIMP_PDB_DRAWABLE, (gchar *)"drawable", (gchar *)"Input drawable" }
  };

  gimp_install_procedure(
    PLUG_IN_NAME,
    "Apply a custom convolution to the whole image",
    "Apply a custom convolution to the whole image",
    PLUG_IN_AUTHOR,
    PLUG_IN_AUTHOR,
    PLUG_IN_VERSION,
    "Testovací plugin",
    "RGB*, GRAY*",
    GIMP_PLUGIN,
    G_N_ELEMENTS (args), 0,
    args, NULL);
  gimp_plugin_menu_register (PLUG_IN_NAME, "<Image>/Filters");// /Generic
  
}

static void
run (const gchar      *name,
     gint              nparams,
     const GimpParam  *param,
     gint             *nreturn_vals,
     GimpParam       **return_vals)
{
  /* Return values */
  static GimpParam values[1];

  GimpDrawable      *drawable;
  gint sel_x1, sel_y1, sel_x2, sel_y2, w, h, padding;
	PluginData         pd;
  GimpRunMode        run_mode;
  GimpPDBStatusType  status;

  *nreturn_vals = 1;
  *return_vals  = values;

  status = GIMP_PDB_SUCCESS;

  if (param[0].type!= GIMP_PDB_INT32) 
	  status=GIMP_PDB_CALLING_ERROR;
  if (param[2].type!=GIMP_PDB_DRAWABLE)
	  status=GIMP_PDB_CALLING_ERROR;

  run_mode = (GimpRunMode) param[0].data.d_int32;
	gimp_ui_init (PLUG_IN_BINARY, FALSE);
	
	drawable = pd.drawable = gimp_drawable_get(param[2].data.d_drawable);
	gimp_drawable_mask_bounds(drawable->drawable_id, &sel_x1, &sel_y1, &sel_x2, &sel_y2);
	pd.img_width = sel_x2 - sel_x1;
  pd.img_height = sel_y2 - sel_y1;
  printf("width: %i height: %i\n", pd.img_width, pd.img_height);
  pd.img_offset_x = sel_x1;
  pd.img_offset_y = sel_y1;
  pd.img_bpp = gimp_drawable_bpp(drawable->drawable_id);

	dialog(&pd);

  values[0].type = GIMP_PDB_STATUS;
  values[0].data.d_status = status;
  gimp_drawable_detach(drawable);
}
