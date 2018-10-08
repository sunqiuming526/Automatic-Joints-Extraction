/*
Szymon Rusinkiewicz
Princeton University

mesh_view.cc
Simple viewer
*/

#include "TriMesh.h"
#include "TriMesh_algo.h"
#include "XForm.h"
#include "GLCamera.h"
#include "GLManager.h"
#include "ICP.h"
#include "strutil.h"
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#ifdef __APPLE__
# include <GLUT/glut.h>
#else
# include <GL/freeglut.h>
#endif
using namespace std;
using namespace trimesh;


#include "shaders.inc"


// Globals

static const char myname[] = "mesh_view";
vector<TriMesh *> meshes;
vector<xform> xforms;
vector<bool> visible;
vector<string> filenames;
vector<point> cpoints;
TriMesh::BSphere global_bsph;
xform global_xf;
GLManager gl;
GLCamera camera;

int current_mesh = -1;

bool draw_2side = false;
bool draw_edges = false;
bool draw_falsecolor = false;
bool draw_flat = true;
bool draw_index = false;
bool draw_lit = true;
bool draw_meshcolor = true;
bool draw_points = false;
bool draw_shiny = true;
bool white_bg = false;
bool grab_only = false;
bool avoid_tstrips = false;
int point_size = 1, line_width = 1;
// ==================================
int level = 60;
bool DRAW_CHARACTERISTIC = 1;
bool DRAW_LEVEL_SET = 1;
bool DRAW_CENTER = 1;
bool DRAW_JOINT = 0;
// ==================================
std::vector<point> joints;
// Struct
typedef struct{
	int index;
	point pos;
	int neib1 = -1;
	int neib2 = -1;
}reeb_point;

// Class: Vertex Link List: store the vertex position of each level-set-curve
// class Vert_ll
// {
// private:
// 	struct vertex{
// 		int index = 0;
// 		vertex* parent = NULL;
// 		vertex* child = NULL;
// 	};
// 	struct vertex start;
//
// };

typedef struct
{
	int index = 0;
	int degree = 0;
	std::vector<int> adj;
	std::vector<point> pos;
}ReebVertices;

typedef struct
{
	int index = 0;
	int degree = 0;
	std::vector<int> adj;
	std::vector<point> pos;
	std::vector<int> flag;
	std::vector<point> cen;
	std::vector<float> area;
	std::vector<float> peri;
} levelset;

std::vector<levelset> LS(level);



// Signal a redraw
void need_redraw()
{
	glutPostRedisplay();
}


// Update global bounding sphere.
void update_bsph()
{
	point boxmin(1e38f, 1e38f, 1e38f);
	point boxmax(-1e38f, -1e38f, -1e38f);
	bool some_vis = false;
	for (size_t i = 0; i < meshes.size(); i++) {
		if (!visible[i])
			continue;
		some_vis = true;
		point c = xforms[i] * meshes[i]->bsphere.center;
		float r = meshes[i]->bsphere.r;
		for (int j = 0; j < 3; j++) {
			boxmin[j] = min(boxmin[j], c[j]-r);
			boxmax[j] = max(boxmax[j], c[j]+r);
		}
	}
	if (!some_vis)
		return;
	point &gc = global_bsph.center;
	float &gr = global_bsph.r;
	gc = 0.5f * (boxmin + boxmax);
	gr = 0.0f;
	for (size_t i = 0; i < meshes.size(); i++) {
		if (!visible[i])
			continue;
		point c = xforms[i] * meshes[i]->bsphere.center;
		float r = meshes[i]->bsphere.r;
		gr = max(gr, dist(c, gc) + r);
	}
}


// Handle auto-spin
bool autospin()
{
	xform tmp_xf = global_xf;
	if (current_mesh >= 0)
		tmp_xf = global_xf * xforms[current_mesh];

	if (!camera.autospin(tmp_xf))
		return false;

	if (current_mesh >= 0) {
		xforms[current_mesh] = inv(global_xf) * tmp_xf;
		update_bsph();
	} else {
		global_xf = tmp_xf;
	}
	return true;
}


// Initialization performed once we have an OpenGL context
void initGL()
{
	gl.make_shader("unlit", unlit_vert, unlit_frag);
	gl.make_shader("phong", phong_vert, phong_frag);
	gl.make_shader("flat", flat_vert, flat_frag);
	if (gl.slow_tstrips())
		avoid_tstrips = true;
}


// Clear the screen
void cls()
{
	glDisable(GL_DITHER);
	glDisable(GL_BLEND);
	glDisable(GL_LIGHTING);
	glDisable(GL_COLOR_MATERIAL);
	if (draw_index)
		glClearColor(0, 0, 0, 0);
	else if (white_bg)
		glClearColor(1, 1, 1, 0);
	else
		glClearColor(0.08f, 0.08f, 0.08f, 0);
	glClearDepth(1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
}


// Return the color to be used in an ID reference image for mesh i
Color index_color(int i)
{
	int steps_per_channel = int(ceil(cbrt(float(meshes.size() + 1))));
	float scale = 1.0f / (steps_per_channel + 1);
	Color c;
	c[0] = scale * ((i + 1) % steps_per_channel);
	c[1] = scale * (((i + 1) / steps_per_channel) % steps_per_channel);
	c[2] = scale * ((i + 1) / sqr(steps_per_channel));
	return c;
}


// Return the false color to be used for mesh i
Color false_color(int i)
{
	return Color::hsv(-3.88f * i, 0.6f + 0.2f * sin(0.42f * i), 1);
}


// Set up color/materials and lights
void setup_color_and_lighting(int i)
{
	Color c(1.0f);
	if (draw_index)
	{
		c = index_color(i);
	}
	else if (draw_falsecolor)
		c = false_color(i);
	gl.color3fv(c);

	if (draw_index || !draw_lit) {
		glDisable(GL_LIGHTING);
		return;
	}

	GLfloat mat_specular[4] = { 0.18f, 0.18f, 0.18f, 0.18f };
	if (!draw_shiny) {
		mat_specular[0] = mat_specular[1] =
		mat_specular[2] = mat_specular[3] = 0.0f;
	}
	GLfloat mat_shininess[] = { 64 };
	GLfloat global_ambient[] = { 0, 0, 0, 0 };
	GLfloat light0_ambient[] = { 0.04f, 0.04f, 0.06f, 0.05f };
	GLfloat light0_diffuse[] = { 0.85f, 0.85f, 0.8f, 0.85f };
	if (current_mesh >= 0 && i != current_mesh) {
		light0_diffuse[0] *= 0.5f;
		light0_diffuse[1] *= 0.5f;
		light0_diffuse[2] *= 0.5f;
	}
	GLfloat light0_specular[] = { 0.85f, 0.85f, 0.85f, 0.85f };
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, draw_2side);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_NORMALIZE);
}


// Draw a mesh
void draw_mesh(int i)
{
	TriMesh *mesh = meshes[i];
	if (!draw_points && avoid_tstrips)
		mesh->need_faces();

	// Transform
	glPushMatrix();
	glMultMatrixd(xforms[i]);

	// Backface culling
	if (draw_2side) {
		glDisable(GL_CULL_FACE);
	} else {
		glCullFace(GL_BACK);
		glEnable(GL_CULL_FACE);
	}

	// Figure out options and passes
	bool unlit = (!draw_lit || draw_index);
	bool phong = (!unlit && !draw_flat);
	bool flat = (!unlit && !phong);

	bool have_faces = avoid_tstrips ? !mesh->faces.empty() :
		!mesh->tstrips.empty();
	bool have_colors = !mesh->colors.empty();

	bool points_pass = (draw_points || !have_faces);
	bool faces_pass = !points_pass;
	bool edges_pass = (faces_pass && !draw_index && draw_edges);

	bool meshcolor = (!draw_index && !draw_falsecolor &&
		have_colors && draw_meshcolor);

	// Activate shader and bind data
	if (unlit)
		gl.use_shader("unlit");
	else if (phong)
		gl.use_shader("phong");
	else // if (flat)
		gl.use_shader("flat");

	setup_color_and_lighting(i);

	unsigned buf;
	if (!(buf = gl.buffer(mesh->vertices)))
		buf = gl.make_buffer(mesh->vertices);
	gl.vertexarray3fv(buf);

	if (phong) {
		if (!(buf = gl.buffer(mesh->normals)))
			buf = gl.make_buffer(mesh->normals);
		gl.normalarray3fv(buf);
	}

	if (meshcolor) {
		if (!(buf = gl.buffer(mesh->colors)))
			buf = gl.make_buffer(mesh->colors);
		gl.colorarray3fv(buf);
	}

	// Drawing passes
	if (points_pass) {
		glPointSize(float(point_size));
		gl.draw_points(mesh->vertices.size());
	}

	if (edges_pass) {
		glPolygonOffset(2.0f, 2.0f);
		glEnable(GL_POLYGON_OFFSET_FILL);
	}

	if (faces_pass) {
		if (avoid_tstrips) {
			if (!(buf = gl.ibuffer(mesh->faces)))
				buf = gl.make_ibuffer(mesh->faces);
			gl.use_ibuffer(buf);
			gl.draw_tris(mesh->faces.size());
		} else {
			if (!(buf = gl.ibuffer(mesh->tstrips)))
				buf = gl.make_ibuffer(mesh->tstrips);
			gl.use_ibuffer(buf);
			gl.draw_tstrips(mesh->tstrips.size());
		}
	}

	if (edges_pass) {
		glDisable(GL_POLYGON_OFFSET_FILL);
		if (flat) {
			gl.use_shader("phong");
			gl.vertexarray3fv(gl.buffer(mesh->vertices));
			if (!gl.buffer(mesh->normals))
				gl.make_buffer(mesh->normals);
			gl.normalarray3fv(gl.buffer(mesh->normals));
		}
		gl.color3f(0, 0, 1.0);
		glPolygonMode(GL_FRONT, GL_LINE);
		glLineWidth(float(line_width));
		if (avoid_tstrips) {
			if (!(buf = gl.ibuffer(mesh->faces)))
				buf = gl.make_ibuffer(mesh->faces);
			gl.use_ibuffer(buf);
			gl.draw_tris(mesh->faces.size());
		} else {
			if (!(buf = gl.ibuffer(mesh->tstrips)))
				buf = gl.make_ibuffer(mesh->tstrips);
			gl.use_ibuffer(buf);
			gl.draw_tstrips(mesh->tstrips.size());
		}
		glPolygonMode(GL_FRONT, GL_FILL);
	}
	glPopMatrix();
	gl.clear_attributes();
}


// Draw the scene
void redraw()
{
	static bool first_time = true;
	if (first_time) {
		initGL();
		first_time = false;
	}

	static bool starting_up = true;
	if (starting_up) {
		// Turn on 1M tris at a time to avoid wait for VBOs
		size_t turned_on = 0;
		size_t i = 0;
		while (i < visible.size()) {
			if (!visible[i]) {
				visible[i] = true;
				void resetview();
				resetview();
				void update_title();
				update_title();
				need_redraw();
				turned_on += meshes[i]->vertices.size();
				if (turned_on >= 1000000)
					break;
			}
			i++;
		}
		if (i == visible.size())
			starting_up = false;
	}

	timestamp t = now();

	camera.setupGL(global_xf * global_bsph.center, global_bsph.r);
	glPushMatrix();
	glMultMatrixd(global_xf);
	cls();

	for (size_t i = 0; i < meshes.size(); i++) {
		if (!visible[i])
			continue;
		draw_mesh(i);
	}
	// ===== Try to draw the characteristic points =========
	if (DRAW_CHARACTERISTIC)
	{
		glPushMatrix();
		glPointSize(10.0f);    //修改点的尺寸，默认大小为1.0f
		glBegin(GL_POINTS);
		// glColor3f(1.0f, 0.0f, 0.0f);
		// glVertex3f(meshes[0]->vertices[0][0], meshes[0]->vertices[0][1], meshes[0]->vertices[0][2]);
		glColor3f(1.0f, 0.0f, 0.0f);

		for (uint i = 0; i < cpoints.size(); i++)
		{
		 glVertex3f(cpoints[i][0],cpoints[i][1],cpoints[i][2]);
		}
		glEnd();
		glPopMatrix();
	}

	// =====================================================
	// Try to draw the Level Set Curve
	if (DRAW_LEVEL_SET)
	{
		glPushMatrix();
		glPointSize(2.0f);
		glBegin(GL_POINTS);
		glColor3f(0.0f, 1.0f, 0.0f);
		for (int i = 0; i < level; i++)
		{
			for (int j = 0; j < 500; j++)
			{
				point tmp = LS[i].pos[j];
				glVertex3f(tmp[0], tmp[1], tmp[2]);
			}
		}
		glEnd();
		glPopMatrix();
	}

	// =====================================================
	// Try to draw the Center Points
	if (DRAW_CENTER)
	{
		glPushMatrix();
		glPointSize(5.0f);
		glBegin(GL_POINTS);
		glColor3f(0.0f, 0.0f, 1.0f);
		for (int i = 0; i < level; i++)
		{
			for (uint j = 0; j < LS[i].cen.size(); j++ )
				glVertex3f(LS[i].cen[j][0], LS[i].cen[j][1], LS[i].cen[j][2]);
		}
		glEnd();
		glPopMatrix();
	}
	// ===========================================================
	// Try to draw the Center Points
	if (DRAW_JOINT)
	{
		glPushMatrix();
		glPointSize(10.0f);
		glBegin(GL_POINTS);
		glColor3f(1.0f, 1.0f, 0.0f);
		for (int i = 0; i < joints.size(); i++)
		{
				glVertex3f(joints[i][0], joints[i][1], joints[i][2]);
		}
		glEnd();
		glPopMatrix();
	}
	// ===========================================================
	glPopMatrix();

	glutSwapBuffers();
	if (grab_only) {
		void dump_image();
		dump_image();
		exit(0);
	}

	glFinish(); // For timing...
	printf("\r                        \r%.1f msec.", 1000.0f * (now() - t));
	fflush(stdout);

	if (autospin())
		need_redraw();
}


// Set the view...
void resetview()
{
	camera.stopspin();

	// Reload mesh xforms
	for (size_t i = 0; i < meshes.size(); i++)
		if (!xforms[i].read(xfname(filenames[i])))
			xforms[i] = xform();

	update_bsph();

	// Set camera to first ".camxf" if we have it...
	for (size_t i = 0; i < filenames.size(); i++) {
		if (global_xf.read(replace_ext(filenames[i], "camxf")))
			return;
	}

	// else default view
	global_xf = xform::trans(0, 0, -5.0f * global_bsph.r) *
	            xform::trans(-global_bsph.center);
}


#define safe_strcat(dst, src) strncat(dst, src, sizeof(dst) - strlen(dst) - 1)


// Set the window title
void update_title()
{
	char title[BUFSIZ/2];
	title[0] = '\0';

	// First the current mesh, if any
	if (current_mesh >= 0) {
		title[0] = '*';
		title[1] = '\0';
		safe_strcat(title, filenames[current_mesh].c_str());
		safe_strcat(title, "* ");
	}

	// Then the other visible meshes
	int nmeshes = meshes.size();
	for (int i = 0; i < nmeshes; i++) {
		if (i == current_mesh || !visible[i])
			continue;
		if (strlen(title) + strlen(filenames[i].c_str()) >
		    sizeof(title) - 2)
			break;
		safe_strcat(title, filenames[i].c_str());
		safe_strcat(title, " ");
	}

	// If nothing visible, just my name
	if (title[0] == '\0')
		safe_strcat(title, myname);

	glutSetWindowTitle(title);
}


// Make some mesh current
void set_current(int i)
{
	camera.stopspin();
	if (i >= 0 && i < (int)meshes.size() && visible[i])
		current_mesh = i;
	else
		current_mesh = -1;
	need_redraw();
	update_title();
}


// Make all meshes visible
void vis_all()
{
	for (size_t i = 0; i < meshes.size(); i++)
		visible[i] = true;
	update_bsph();
	need_redraw();
	update_title();
}


// Hide all meshes
void vis_none()
{
	for (size_t i = 0; i < meshes.size(); i++)
		visible[i] = false;
	current_mesh = -1;
	update_bsph();
	need_redraw();
	update_title();
}


// Make the "previous" or "next" mesh visible
void vis_prev()
{
	// Find the first visible mesh
	int first_vis = -1;
	for (int i = meshes.size() - 1; i >= 0; i--) {
		if (visible[i])
			first_vis = i;
	}
	if (first_vis < 0)
		first_vis = 0;

	// Now find the previous one
	int prev_vis = (first_vis + meshes.size() - 1) % meshes.size();

	// Now make only that one visible
	for (size_t i = 0; i < meshes.size(); i++)
		visible[i] = (int(i) == prev_vis);

	current_mesh = -1;
	update_bsph();
	need_redraw();
	update_title();
}

void vis_next()
{
	// Find the last visible mesh
	int last_vis = -1;
	for (size_t i = 0; i < meshes.size(); i++) {
		if (visible[i])
			last_vis = i;
	}
	if (last_vis < 0)
		last_vis = meshes.size() - 1;

	// Now find the next one
	int next_vis = (last_vis + 1) % meshes.size();

	// Now make only that one visible
	for (size_t i = 0; i < meshes.size(); i++)
		visible[i] = (int(i) == next_vis);

	current_mesh = -1;
	update_bsph();
	need_redraw();
	update_title();
}


// Change visiblility of a mesh
void toggle_vis(int i)
{
	if (i >= 0 && i < (int)meshes.size())
		visible[i] = !visible[i];
	if (current_mesh == i && !visible[i])
		set_current(-1);
	update_bsph();
	need_redraw();
	update_title();
}


// Save the current image to a PPM file.
// Uses the next available filename matching filenamepattern
void dump_image()
{
	// Find first non-used filename
	const char filenamepattern[] = "img%d.ppm";
	int imgnum = 0;
	FILE *f;
	while (1) {
		char filename[BUFSIZ];
		sprintf(filename, filenamepattern, imgnum++);
		f = fopen(filename, "rb");
		if (!f) {
			f = fopen(filename, "wb");
			printf("\n\nSaving image %s... ", filename);
			fflush(stdout);
			break;
		}
		fclose(f);
	}

	// Read pixels
	GLint V[4];
	glGetIntegerv(GL_VIEWPORT, V);
	GLint width = V[2], height = V[3];
	char *buf = new char[width*height*3];
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(V[0], V[1], width, height, GL_RGB, GL_UNSIGNED_BYTE, buf);

	// Flip top-to-bottom
	for (int i = 0; i < height/2; i++) {
		char *row1 = buf + 3 * width * i;
		char *row2 = buf + 3 * width * (height - 1 - i);
		for (int j = 0; j < 3 * width; j++)
			swap(row1[j], row2[j]);
	}

	// Write out file
	fprintf(f, "P6\n#\n%d %d\n255\n", width, height);
	fwrite(buf, width*height*3, 1, f);
	fclose(f);
	delete [] buf;

	printf("Done.\n\n");
}


// Save scan transforms
void save_xforms()
{
	for (size_t i = 0; i < xforms.size(); i++) {
		string xffile = xfname(filenames[i]);
		printf("Writing %s\n", xffile.c_str());
		xforms[i].write(xffile);
	}
}


// Save camera xform
void save_cam_xform()
{
	std::string camfile = replace_ext(filenames[0], "camxf");
	printf("Writing %s\n", camfile.c_str());
	global_xf.write(camfile);
}


// ICP
void do_icp(int n)
{
	camera.stopspin();

	if (current_mesh < 0 || current_mesh >= (int)meshes.size())
		return;
	if (n < 0 || n >= (int)meshes.size())
		return;
	if (!visible[n] || !visible[current_mesh] || n == current_mesh)
		return;
	ICP(meshes[n], meshes[current_mesh], xforms[n], xforms[current_mesh], 2);
	update_bsph();
	need_redraw();
}


// Handle mouse button and motion events
static unsigned buttonstate = 0;

void doubleclick(int button, int x, int y)
{
	// Render and read back ID reference image
	camera.setupGL(global_xf * global_bsph.center, global_bsph.r);
	glPushMatrix();
	glMultMatrixd(global_xf);

	draw_index = true;
	cls();
	for (size_t i = 0; i < meshes.size(); i++) {
		if (!visible[i])
			continue;
		draw_mesh(i);
	}
	draw_index = false;

	glPopMatrix();

	GLint V[4];
	glGetIntegerv(GL_VIEWPORT, V);
	y = int(V[1] + V[3]) - 1 - y;
	unsigned char pix[3];
	glReadPixels(x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, pix);

	// Find closest color
	Color pix_color(pix);
	int clicked_mesh = -1;
	float clicked_dist2 = dist2(pix_color, Color());
	for (size_t i = 0; i < meshes.size(); i++) {
		Color mesh_color = index_color(i);
		float mesh_dist2 = dist2(pix_color, mesh_color);
		if (mesh_dist2 < clicked_dist2) {
			clicked_mesh = i;
			clicked_dist2 = mesh_dist2;
		}
	}

	if (button == 0 || buttonstate == (1 << 0)) {
		// Double left click - select a mesh
		set_current(clicked_mesh);
	} else if (button == 2 || buttonstate == (1 << 2)) {
		// Double right click - ICP current to clicked-on
		do_icp(clicked_mesh);
	}
}

void mousehelperfunc(int x, int y)
{
	static const Mouse::button physical_to_logical_map[] = {
		Mouse::NONE,  Mouse::ROTATE, Mouse::MOVEXY, Mouse::MOVEZ,
		Mouse::MOVEZ, Mouse::MOVEXY, Mouse::MOVEXY, Mouse::MOVEXY,
	};

	Mouse::button b = Mouse::NONE;
	if (buttonstate & (1 << 3))
		b = Mouse::WHEELUP;
	else if (buttonstate & (1 << 4))
		b = Mouse::WHEELDOWN;
	else if (buttonstate & (1 << 30))
		b = Mouse::LIGHT;
	else
		b = physical_to_logical_map[buttonstate & 7];

	if (current_mesh < 0) {
		camera.mouse(x, y, b,
		             global_xf * global_bsph.center, global_bsph.r,
		             global_xf);
	} else {
		xform tmp_xf = global_xf * xforms[current_mesh];
		camera.mouse(x, y, b,
		             tmp_xf * meshes[current_mesh]->bsphere.center,
		             meshes[current_mesh]->bsphere.r,
		             tmp_xf);
		xforms[current_mesh] = inv(global_xf) * tmp_xf;
		update_bsph();
	}
}

void mousemotionfunc(int x, int y)
{
	mousehelperfunc(x,y);
	if (buttonstate)
		need_redraw();
}

void mousebuttonfunc(int button, int state, int x, int y)
{
	static timestamp last_click_time;
	static unsigned last_click_buttonstate = 0;
	static float doubleclick_threshold = 0.4f;

	if (glutGetModifiers() & GLUT_ACTIVE_CTRL)
		buttonstate |= (1 << 30);
	else
		buttonstate &= ~(1 << 30);

	if (button == 5 || button == 7) {
		if (state == GLUT_DOWN)
			vis_prev();
		return;
	} else if (button == 6 || button == 8) {
		if (state == GLUT_DOWN)
			vis_next();
		return;
	}

	if (state == GLUT_DOWN) {
		buttonstate |= (1 << button);
		if (buttonstate == last_click_buttonstate &&
		    now() - last_click_time < doubleclick_threshold &&
		    button < 3) {
			doubleclick(button, x, y);
			last_click_buttonstate = 0;
		} else {
			last_click_time = now();
			last_click_buttonstate = buttonstate;
		}
	} else {
		buttonstate &= ~(1 << button);
	}

	mousehelperfunc(x, y);
	if (buttonstate & ((1 << 3) | (1 << 4))) // Wheel
		need_redraw();
	if ((buttonstate & 7) && (buttonstate & (1 << 30))) // Light
		need_redraw();
	if (autospin()) {
		last_click_buttonstate = 0;
		need_redraw();
	}
}


// Keyboard
#define Ctrl (1-'a')
void keyboardfunc(unsigned char key, int, int)
{
	switch (key) {
		case ' ':
			if (current_mesh < 0)
				resetview();
			else
				set_current(-1);
			break;
		case '@': // Shift-2
			draw_2side = !draw_2side; break;
		case 'c':
			draw_meshcolor = !draw_meshcolor; break;
		case 'e':
			draw_edges = !draw_edges; break;
		case 'f':
			draw_falsecolor = !draw_falsecolor; break;
		case 'F':
			draw_flat = !draw_flat; break;
		case 'l':
			draw_lit = !draw_lit; break;
		case 'p':
			point_size++; break;
		case 'P':
			point_size = max(1, point_size - 1); break;
		case Ctrl+'p':
			draw_points = !draw_points; break;
		case 's':
			draw_shiny = !draw_shiny; break;
		case 't':
			line_width++; break;
		case 'T':
			line_width = max(1, line_width - 1); break;
		case 'w':
			white_bg = !white_bg; break;
		case 'I':
			dump_image(); break;
		case Ctrl+'x':
			save_xforms();
			break;
		case Ctrl+'v':
			save_cam_xform();
			break;
		case '\033': // Esc
		case Ctrl+'c':
		case Ctrl+'q':
		case 'Q':
		case 'q':
			exit(0);
		case Ctrl+'a':
			vis_all(); break;
		case Ctrl+'n':
			vis_none(); break;
		case ',':
			vis_prev(); break;
		case '.':
			vis_next(); break;
		case '1': case '2': case '3': case '4': case '5':
		case '6': case '7': case '8': case '9':
			toggle_vis(key - '1'); break;
		case '0':
			toggle_vis(9); break;
		case '-':
			toggle_vis(10); break;
		case '=':
			toggle_vis(11); break;
	}
	need_redraw();
}


void usage()
{
	fprintf(stderr, "Usage: %s [-grab] infile...\n", myname);
	exit(1);
}
float dist(TriMesh* themesh, int i, int j)
{
	return sqrt(pow(themesh->vertices[i][0] - themesh->vertices[j][0], 2)
						+ pow(themesh->vertices[i][1] - themesh->vertices[j][1], 2)
						+ pow(themesh->vertices[i][2] - themesh->vertices[j][2], 2));
}

float peri_dist(int x, int y, int i)
{
	return sqrt(pow(LS[i].pos[x][0] - LS[i].pos[y][0], 2)
						+ pow(LS[i].pos[x][1] - LS[i].pos[y][1], 2)
						+ pow(LS[i].pos[x][2] - LS[i].pos[y][2], 2));
}

float circle_like(float area, float peri)
{
	return (4 * 3.14 * area / (peri * peri));
}
int DJ(TriMesh* themesh, int start, float** dist_mat, float* sum_dist, bool M, float* morse = NULL)
{
	// Initialization

	int vertex_max = themesh->vertices.size();
	// int face_max = themesh->faces.size();

	bool flag[vertex_max] = {0} ;

	float dist[vertex_max] = {999};
	// =====================================================
	for (int s = 0; s < vertex_max; s++)
	{
		// U.push_back(it);
		dist[s] = dist_mat[start][s];
	}
	dist[start] = 0;
	flag[start] = 1;


	//===================================================================================
	// Update the distant
	float min;
	float tmp;
	int k = 0;

	for (int i = 1; i < vertex_max; i++)
	{
		min = 999;
		// find the point of the shortest path
		for (int j = 0; j < vertex_max; j++)
    {

        if (flag[j] == 0 && dist[j] < min)
        {
            min = dist[j];
            k = j;
        }
    }
		// std::cout << k << std::endl;

		flag[k] = 1; // K is of the shortest path

		// update the remaining distant

		for (int j = 0; j < vertex_max; j++)
    {
      tmp = (dist_mat[k][j]==999 ? 999 : (min + dist_mat[k][j])); // 防止溢出
			// if(j == 671 && dist_mat[k][j]!=999)
				// std::cout << "111 " << dist_mat[k][j] <<" " << tmp << std::endl;
      if (flag[j] == 0 && (tmp  < dist[j]) )
      {
          dist[j] = tmp;

      }
    }
	}
	// sum_dist is the sum of distance of point_i to all the points in set V, where V is the characteristic points set.
	for (int i = 0; i < vertex_max; i++)
	{
		sum_dist[i] += dist[i];
	}

	// if the source point is head, then set the morse function
	if (M == true)
		for (int i = 0; i < vertex_max; i++)
			morse[i] = dist[i];

	// return the point of the max distance
	float dmax = 0;
	int imax = 0;
  for (int i = 0; i < vertex_max; i++)
		{
			if (sum_dist[i] > dmax)
			{
				dmax = sum_dist[i];
				imax = i;
			}
		}


	cpoints.push_back(themesh->vertices[imax]);
	return imax;

	// std::cout << "dist: " << dist[1] << std::endl;
}

int check(std::vector<std::vector<int>> v, int a, int b)
{
	vector<std::vector<int>>::iterator it;
	int count = 0;
	for (it = v.begin(); it != v.end(); it++)
	{
		if(((*it)[0] == a && (*it)[1] == b) || ((*it)[0] == b && (*it)[1] == a))
			return count;
		count++;
	}
	return -1;
}

float heron(int x, int y, int z, int i)
{
	float l1 = sqrt(pow(LS[i].pos[x][0] - LS[i].pos[y][0], 2)
						+ pow(LS[i].pos[x][1] - LS[i].pos[y][1], 2)
						+ pow(LS[i].pos[x][2] - LS[i].pos[y][2], 2));
	float l2 = sqrt(pow(LS[i].pos[x][0] - LS[i].pos[z][0], 2)
						+ pow(LS[i].pos[x][1] - LS[i].pos[z][1], 2)
						+ pow(LS[i].pos[x][2] - LS[i].pos[z][2], 2));
	float l3 = sqrt(pow(LS[i].pos[z][0] - LS[i].pos[y][0], 2)
						+ pow(LS[i].pos[z][1] - LS[i].pos[y][1], 2)
						+ pow(LS[i].pos[z][2] - LS[i].pos[y][2], 2));
	float s = 0.5 * (l1 + l2 + l3);
	float S = pow(s * (s - l1) * (s - l2) * (s - l3), 0.5);
	return S;
}
int main(int argc, char *argv[])
{
	unsigned initDisplayMode = GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH;

#ifdef USE_CORE_PROFILE
# ifdef __APPLE__
	initDisplayMode |= GLUT_3_2_CORE_PROFILE;
# else
	glutInitContextVersion(3, 2);
	glutInitContextProfile(GLUT_CORE_PROFILE);
# endif
#endif
	glutInitDisplayMode(initDisplayMode);

	// We'd like to set this based on GLUT_SCREEN_{WIDTH|HEIGHT}
	// but we can't do that until after glutInit().
	// So, we set this to some value here...
	glutInitWindowSize(1, 1);
	glutInit(&argc, argv);

	// ... and if it hasn't been changed by command-line arguments
	// we'll set it to what we really want
	if (glutGet(GLUT_INIT_WINDOW_WIDTH) == 1) {
		int window_size = min(glutGet(GLUT_SCREEN_WIDTH),
			glutGet(GLUT_SCREEN_HEIGHT)) * 3 / 4;
		glutInitWindowSize(window_size, window_size);
	}

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-grab")) {
			grab_only = true;
			continue;
		}
		const char *filename = argv[i];
		TriMesh *themesh = TriMesh::read(filename);
		if (!themesh)
			usage();
		meshes.push_back(themesh);
		xforms.push_back(xform());
		visible.push_back(false);
		filenames.push_back(filename);
	}
	if (meshes.empty())
		usage();

// #pragma omp parallel for
	for (size_t i = 0; i < meshes.size(); i++) {
		meshes[i]->need_tstrips();
		// meshes[i]->clear_grid();
		// meshes[i]->clear_faces();
		// meshes[i]->clear_neighbors();
		// meshes[i]->clear_adjacentfaces();
		// meshes[i]->clear_across_edge();
		// reorder_verts(meshes[i]);
		meshes[i]->need_normals(true);
		meshes[i]->need_bsphere();
		meshes[i]->convert_strips(TriMesh::TSTRIP_TERM);

		// If any loaded mesh is big, enable Phong shading
		if (meshes[i]->vertices.size() > 240)
			draw_flat = false;
	}

// ================================================
// 										MY PART
	const int vertex_max = meshes[0]->vertices.size();
	const int face_max = meshes[0]->faces.size();
	float** adjacent_mat = new float*[vertex_max]();
	float sum_dist[vertex_max] = {0};
	float morse[vertex_max] = {0};
	for (int i = 0; i < level; i++)
		{
			LS[i].adj.resize(1000);
			LS[i].pos.resize(1000);
		}
	for (int i = 0; i < vertex_max; i++)
	{
		adjacent_mat[i] = new float[vertex_max]();
	}


	// Initialize the adjacent_mat
	// set the origin value to 999
	for (int i = 0; i < vertex_max; i++)
	{
		for (int j = 0; j < vertex_max; j++)
		{
			adjacent_mat[i][j] = 999;
		}
	}
	// update the connected edges
	for(int i = 0; i < face_max; i++)
	{
		int a = meshes[0]->faces[i][0];
		int b = meshes[0]->faces[i][1];
		int c = meshes[0]->faces[i][2];
		// std::cout << meshes[0]->faces[i] << std::endl;

		adjacent_mat[a][b] = dist(meshes[0], a, b);
		adjacent_mat[b][a] = dist(meshes[0], a, b);

		adjacent_mat[a][c] = dist(meshes[0], a, c);
		adjacent_mat[c][a] = dist(meshes[0], a, c);

		adjacent_mat[b][c] = dist(meshes[0], c, b);
		adjacent_mat[c][b] = dist(meshes[0], c, b);
	}

	// ===== Get 5 characteristic points by Dijkstra ======
	int last = 0;
	last = DJ(meshes[0], 0, adjacent_mat, sum_dist, false);
	// source point is the head
	last = DJ(meshes[0], last, adjacent_mat, sum_dist, true, morse);
	float amax = morse[last]; // amax the largest distance to the head point
	last = DJ(meshes[0], last, adjacent_mat, sum_dist, true, morse);
	float L = morse[last];
	last = DJ(meshes[0], last, adjacent_mat, sum_dist, false);
	last = DJ(meshes[0], last, adjacent_mat, sum_dist, false);
	// =====================================================
	// for(int i =0; i < vertex_max; i++)
	// 	std::cout << morse[i] << std::endl;
	// Set the distance step
	// int level = level;
	float step = L / level;
	// std::vector<levelset> LS;
	int a, b, c;

	for (int i = 1; i < level; i++)
	{
		 // std::cout << "check level: " << i << std::endl;
		// should be declared in/out of the for loop ???
		std::vector<std::vector<int>> edge_process;
		std::vector<reeb_point> RP;

		levelset ls;
		ls.index = i;
		float bi = step * i;
		for (int j = 0; j < face_max; j++)
		{
			// std::cout << "face level: " << j << std::endl;
			a = meshes[0]->faces[j][0];
			b = meshes[0]->faces[j][1];
			c = meshes[0]->faces[j][2];
			// z => far
			int tmp = morse[a] > morse[b] ? a:b;
			int z = morse[tmp] > morse[c] ? tmp:c;
			// x => near
			tmp = morse[a] < morse[b] ? a:b;
			int x = morse[tmp] < morse[c] ? tmp:c;
			// y => middle
			int y = a + b + c - z - x;

			// std::cout << x << " " << y << " " << z << std::endl;
			// std::cout << morse[x] << " " << morse[y] << " " << morse[z] << "bi: " << bi << std::endl;

			//  ====================
			// || Regular triangle ||
			//  ====================
			if (morse[x] < bi && morse[y] > bi && morse[z] > bi)
			{
				// first check whether the edge has been processed
				reeb_point new_rp1, new_rp2;
				int chc1 = check(edge_process, x, y); // For a processed edge, chc1 is the index
				if (chc1 == -1) // the edge has not been processed => calculate the position and index
				{
					point pc = (bi - morse[x]) / (morse[y] - morse[x])
										* (meshes[0]->vertices[y] - meshes[0]->vertices[x])
										+ meshes[0]->vertices[x];
					std::vector<int> edge1;
					edge1.push_back(x);
					edge1.push_back(y);
					edge_process.push_back(edge1);
					new_rp1.pos = pc;
					// the point index is the same as the corresponding edge index
					new_rp1.index = edge_process.size() - 1;
				}
				else
					new_rp1 = RP[chc1]; // retrieve the point in the RP(reeb points)

				// chc2 is the same as chc1 above
				int chc2 = check(edge_process, x, z);
				if (chc2 == -1)
				{
					point pd = (bi - morse[x]) / (morse[z] - morse[x])
										* (meshes[0]->vertices[z] - meshes[0]->vertices[x])
										+ meshes[0]->vertices[x];
					std::vector<int> edge2;
					edge2.push_back(x);
					edge2.push_back(z);
					edge_process.push_back(edge2);
					new_rp2.pos = pd;
					new_rp2.index = edge_process.size() - 1;
				}
				else
					new_rp2 = RP[chc2];
				// std::cout << "chc: " << chc1 << ", " << chc2 << std::endl;
				 if (chc1 == -1)
				 {
				   new_rp1.neib1 = new_rp2.index;
					 RP.push_back(new_rp1);
					 // std::cout << new_rp1.index << " +neib: " << new_rp1.neib1 << std::endl;
				 }
				 else
				 {
					 RP[chc1].neib2 = new_rp2.index;
					  // std::cout << new_rp1.index << " +neib: " << new_rp1.neib1 << std::endl;
				 }
				 if (chc2 == -1)
				 {
					 new_rp2.neib1 = new_rp1.index;
					 RP.push_back(new_rp2);
					 // std::cout << new_rp2.index << " +neib: " << new_rp2.neib1 << std::endl;
				 }
				 else
				 {
 						RP[chc2].neib2 = new_rp1.index;
						// std::cout << new_rp2.index << " +neib: " << new_rp2.neib1 << std::endl;
				}

			}

			//  =====================
			// || Inverted triangle ||
			//  =====================
			else if (morse[x] < bi && morse[y] < bi && morse[z] > bi)
			{
				// first check whether the edge has been processed
				reeb_point new_rp1, new_rp2;
				int chc1 = check(edge_process, y, z); // For a processed edge, chc1 is the index
				if (chc1 == -1)
				{
					point pc = (bi - morse[y]) / (morse[z] - morse[y])
										* (meshes[0]->vertices[z] - meshes[0]->vertices[y])
										+ meshes[0]->vertices[y];
					std::vector<int> edge1;
					edge1.push_back(y);
					edge1.push_back(z);
					edge_process.push_back(edge1);
					new_rp1.pos = pc;
					new_rp1.index = edge_process.size() - 1;
				}
				else
					new_rp1 = RP[chc1];

				int chc2 = check(edge_process, x, z);
				if (chc2 == -1)
				{
					point pd = (bi - morse[x]) / (morse[z] - morse[x])
										* (meshes[0]->vertices[z] - meshes[0]->vertices[x])
										+ meshes[0]->vertices[x];
					std::vector<int> edge2;
					edge2.push_back(x);
					edge2.push_back(z);
					edge_process.push_back(edge2);
					new_rp2.pos = pd;
					new_rp2.index = edge_process.size() - 1;
				}
				else
					new_rp2 = RP[chc2];

				// update the connected points in RP
				if (chc1 == -1)
				{
					new_rp1.neib1 = new_rp2.index;
					RP.push_back(new_rp1);
					// std::cout << new_rp1.index << " +neib: " << new_rp1.neib1 << std::endl;
				}
				else
				{
					RP[chc1].neib2 = new_rp2.index;
					 // std::cout << new_rp1.index << " +neib: " << new_rp1.neib1 << std::endl;
				}
				if (chc2 == -1)
				{
					new_rp2.neib1 = new_rp1.index;
					RP.push_back(new_rp2);
					// std::cout << new_rp2.index << " +neib: " << new_rp2.neib1 << std::endl;
				}
				else
				{
					 RP[chc2].neib2 = new_rp1.index;
					 // std::cout << new_rp2.index << " +neib: " << new_rp2.neib1 << std::endl;
			 }
		 }
	 }

	 // for (int j = 0; j < RP.size(); j++)
	 // {
		//  std::cout << "index:" << RP[j].index << " neib1: " << RP[j].neib1 << " neib2: " << RP[j].neib2 << std::endl;
	 // }
	 // // return 0;
		LS[i].adj.resize(RP.size());
		LS[i].pos.resize(RP.size());
		LS[i].flag.resize(RP.size());
		// Initialize LS.flag to -1
		for(uint j = 0; j < RP.size(); j++)
		{
			LS[i].flag[j] = -1;
		}
		int start_point = RP[0].index;
		LS[i].index = i;
		LS[i].degree++;
		LS[i].adj[start_point] = RP[0].neib1; // Set the head
		LS[i].pos[start_point] = RP[0].pos;
		LS[i].flag[start_point] = 1;
		int next = RP[0].neib1;
		int prev = start_point;
		bool flag = 1;
		while (flag)
		{
			// Since each point has two neighbors, this part identify which neighbor should be choosed as the next node.
			if(RP[next].neib1 != prev )
			{
				LS[i].adj[next] = RP[next].neib1;
				LS[i].pos[next] = RP[next].pos;
				LS[i].flag[next] = LS[i].degree;
				prev = next;
				next = LS[i].adj[next];
			}
			else
			{
				LS[i].adj[next] = RP[next].neib2;
				LS[i].pos[next] = RP[next].pos;
				LS[i].flag[next] = LS[i].degree;
				prev = next;
				next = LS[i].adj[next];
			}
			if (next == start_point)
			{
				flag = 0;
				for(uint j = 0; j < LS[i].flag.size(); j++)
				{
					if (LS[i].flag[j] == -1)
					{
						flag = 1;
						next = j;
						start_point = j;
						LS[i].degree++;
						break;
					}
				}
			}
		}
		// Calculate the centroid of each layer
		LS[i].cen.resize(LS[i].degree);
		LS[i].area.resize(LS[i].degree);
		LS[i].peri.resize(LS[i].degree);

		for(uint j = 0; j < LS[i].cen.size(); j++)
		{
			int sz = 0;
			for(uint k = 0; k < LS[i].pos.size(); k++)
				if (LS[i].flag[k] == (int)j + 1)
				{
					LS[i].cen[j] += LS[i].pos[k];
					sz++;
				}
			LS[i].cen[j] /= sz;
		}

		// Calculate the area of each level
		for (int j = 0; j < LS[i].degree; j++)
		{
			int k = 0; // k is the start point of each degree
			float area = 0; //
			float peri = 0;
			for (k = 0; k < LS[i].flag.size(); k++)
				if (LS[i].flag[k] == j + 1)
					break;
			// Invoke Heron's Function
			next = LS[i].adj[k];
			peri += peri_dist(k, next, i);

			while (LS[i].adj[next] != k) // If next == k, we have gone through the circle
			{
				int n_next = LS[i].adj[next];
				//海伦公式 => 面积+= =>　next = n_next;
				area += heron(k, next, n_next, i);
				peri += peri_dist(next, n_next, i);
				next = n_next;
			}
			LS[i].area[j] = area;
			peri += peri_dist(next, k, i);
			LS[i].peri[j] = peri;
		}
	}
	// Extract the Neck
	float min_circle = 999;
	point joint_position;
	for (int i = 0; i < 0.2*level; i++)
	{
		for (int j = 0; j < LS[i].degree; j++)
		{
			float circle = circle_like(LS[i].area[j], LS[i].peri[j]);
			if (circle < min_circle)
			{
				min_circle = circle;
				joint_position = LS[i].cen[j];
			}
		}
	}
	joints.push_back(joint_position);

	// Extract the knees
	float min_circle1 = 999;
	float min_circle2 = 999;
	point knee_position1;
	point knee_position2;
	for (int i = 0.6 * level; i < 0.8 * level; i++)
	{
		if (LS[i].degree == 2)
		{
			float circle1 = circle_like(LS[i].area[0], LS[i].peri[0]);
			float circle2 = circle_like(LS[i].area[1], LS[i].peri[1]);
			if (circle1 < min_circle1)
			{
				min_circle1 = circle1;
				knee_position1 = LS[i].cen[0];
			}
			if (circle2 < min_circle2)
			{
				min_circle2 = circle2;
				knee_position2 = LS[i].cen[1];
			}
		}
	}
	joints.push_back(knee_position1);
	joints.push_back(knee_position2);

	// Extract the elbows
	min_circle1 = 999;
	min_circle2 = 999;
	point elbow_position1;
	point elbow_position2;
	int elbow_start = 0;
	int elbow_end = 0;
	// try to find the level of elbows
	for (int i = 0; i < level; i++)
	{
		if (LS[i].degree == 3 && elbow_start == 0)
		{
			elbow_start = i;
		}
		if (LS[i].degree == 1 && elbow_start != 0)
		{
			elbow_end = i;
			break;
		}
	}

	int elbow_lenth = elbow_end - elbow_start;
	for (int i = elbow_start + 0.2 * elbow_lenth; i < elbow_end - 0.6 * elbow_lenth; i++)
	{
		int e1 = 0, e2 = 0;
		if (LS[i].degree == 3)
		{
			// check which degree is trunk
			if (LS[i].area[0] > LS[i].area[1] && LS[i].area[0] > LS[i].area[2])
			{
				e1 = 1;
				e2 = 2;
			}
			else if (LS[i].area[1] > LS[i].area[0] && LS[i].area[1] > LS[i].area[2])
			{
				e1 = 0;
				e2 = 2;
			}
			else
			{
				e1 = 0;
				e2 = 1;
			}
			float circle1 = circle_like(LS[i].area[e1], LS[i].peri[e1]);
			float circle2 = circle_like(LS[i].area[e2], LS[i].peri[e2]);
			if (circle1 < min_circle1)
			{
				min_circle1 = circle1;
				elbow_position1 = LS[i].cen[e1];
			}
			if (circle2 < min_circle2)
			{
				min_circle2 = circle2;
				elbow_position2 = LS[i].cen[e2];
			}
		}
	}
	joints.push_back(elbow_position1);
	joints.push_back(elbow_position2);

	glutCreateWindow(myname);
	glutDisplayFunc(redraw);
	glutMouseFunc(mousebuttonfunc);
	glutMotionFunc(mousemotionfunc);
	glutKeyboardFunc(keyboardfunc);

	resetview();
	update_title();
	glutMainLoop();
}
