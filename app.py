from flask import Flask, render_template, request, jsonify
import math
import numpy as np

app = Flask(__name__)

class Panel:
    def __init__(self, name, height, width, power, cut_length=0, cut_height=0):
        self.name = name
        self.height = height
        self.width = width
        self.power = power
        self.cut_length = cut_length
        self.cut_height = cut_height

PANELS = {
    "A": Panel("A", 312, 300, 4.86),
    "B": Panel("B", 312, 60, 1),
    "C": Panel("C", 52, 300, 1.3),
    "D": Panel("D", 52, 60, 0.32),
}

GRIDFLEX_PANELS = {
    "GridFlex": Panel("GridFlex", 240, 480, 10, 20, 20),
    "GridFlex DW": Panel("GridFlex DW", 240, 480, 20, 24, 24),
    "GridFlex RGBW": Panel("GridFlex RGBW", 305, 610, 40, 152.5, 305/6),
}

def rotate_panel(panel):
    return Panel(panel.name, panel.width, panel.height, panel.power, panel.cut_height, panel.cut_length)

def fit_layout(box_w, box_h, panels, orientation, psu_wattage=None, psu_max_load=0.8, is_gridflex=False):
    if orientation == "Landscape":
        for key in panels:
            panels[key] = rotate_panel(panels[key])
    
    layout = []
    
    if is_gridflex:
        panel = list(panels.values())[0]
        panel_count = 0
        total_power = 0
        
        count_cols = int(box_w // panel.width)
        count_rows = int(box_h // panel.height)
        used_w = count_cols * panel.width
        used_h = count_rows * panel.height
        
        for row in range(count_rows):
            for col in range(count_cols):
                x = col * panel.width
                y = row * panel.height
                layout.append((x, y, panel, panel.width, panel.height))
                panel_count += 1
                total_power += panel.power
        
        remaining_w = box_w - used_w
        if remaining_w >= panel.cut_length:
            cut_width = panel.cut_length * int(remaining_w // panel.cut_length)
            for row in range(max(1, count_rows)):
                x = used_w
                y = row * panel.height
                if y + panel.height <= box_h:
                    layout.append((x, y, panel, cut_width, panel.height))
                    panel_count += 1
                    total_power += panel.power * (cut_width / panel.width) * (panel.height / panel.height)
        
        remaining_h = box_h - used_h
        if remaining_h >= panel.cut_height:
            cut_height = panel.cut_height * int(remaining_h // panel.cut_height)
            for col in range(max(1, count_cols)):
                x = col * panel.width
                y = used_h
                if x + panel.width <= box_w:
                    layout.append((x, y, panel, panel.width, cut_height))
                    panel_count += 1
                    total_power += panel.power * (panel.width / panel.width) * (cut_height / panel.height)
        
        if remaining_w >= panel.cut_length and remaining_h >= panel.cut_height:
            x = used_w
            y = used_h
            layout.append((x, y, panel, cut_width, cut_height))
            panel_count += 1
            total_power += panel.power * (cut_width / panel.width) * (cut_height / panel.height)
        
        total_width_used = used_w + (cut_width if remaining_w >= panel.cut_length else 0)
        total_height_used = used_h + (cut_height if remaining_h >= panel.cut_height else 0)
        horizontal_leftover = max(0, box_w - total_width_used)
        leftover_left = horizontal_leftover / 2
        leftover_right = horizontal_leftover / 2
        vertical_leftover = max(0, box_h - total_height_used)
        leftover_top = vertical_leftover / 2
        leftover_bottom = vertical_leftover / 2
        
        counts = {panel.name: panel_count}
    else:
        A = panels["A"]
        B = panels["B"]
        C = panels["C"]
        D = panels["D"]
        
        count_A_cols = int(box_w // A.width)
        count_A_rows = int(box_h // A.height)
        used_w_A = count_A_cols * A.width
        used_h_A = count_A_rows * A.height
        leftover_h = box_h - used_h_A
        
        count_C_cols = count_A_cols
        count_C_rows = int(leftover_h // C.height) if leftover_h >= C.height else 0
        used_h_C = count_C_rows * C.height
        total_used_h = used_h_A + used_h_C
        
        leftover_w = box_w - used_w_A
        count_B_rows = int(total_used_h // B.height) if total_used_h >= B.height else 0
        count_B_cols = int(leftover_w // B.width) if leftover_w >= B.width else 0
        used_w_B = count_B_cols * B.width
        
        remaining_w = leftover_w - used_w_B
        remaining_h = leftover_h - used_h_C
        count_D_cols = int(remaining_w // D.width) if remaining_w >= D.width else 0
        count_D_rows = int(remaining_h // D.height) if remaining_h >= D.height else 0
        used_w_D = count_D_cols * D.width
        used_h_D = count_D_rows * D.height
        
        total_width_used = used_w_A + used_w_B + used_w_D
        total_height_used = used_h_A + used_h_C + used_h_D
        horizontal_leftover = max(0, box_w - total_width_used)
        leftover_left = horizontal_leftover / 2
        leftover_right = horizontal_leftover / 2
        vertical_leftover = max(0, box_h - total_height_used)
        leftover_top = vertical_leftover / 2
        leftover_bottom = vertical_leftover / 2
        
        if total_width_used > box_w or total_height_used > box_h:
            count_A_cols = int((box_w - used_w_B - used_w_D) // A.width)
            used_w_A = count_A_cols * A.width
            count_A_rows = int((box_h - used_h_C - used_h_D) // A.height)
            used_h_A = count_A_rows * A.height
            total_width_used = used_w_A + used_w_B + used_w_D
            total_height_used = used_h_A + used_h_C + used_h_D
            horizontal_leftover = max(0, box_w - total_width_used)
            leftover_left = horizontal_leftover / 2
            leftover_right = horizontal_leftover / 2
            vertical_leftover = max(0, box_h - total_height_used)
            leftover_top = vertical_leftover / 2
            leftover_bottom = vertical_leftover / 2
        
        for row in range(count_A_rows):
            for col in range(count_A_cols):
                x = leftover_left + col * A.width
                y = leftover_bottom + row * A.height
                if y + A.height <= box_h:
                    layout.append((x, y, A, A.width, A.height))
        
        for row_c in range(count_C_rows):
            for col_c in range(count_C_cols):
                x = leftover_left + col_c * C.width
                y = leftover_bottom + used_h_A + row_c * C.height
                if y + C.height <= box_h:
                    layout.append((x, y, C, C.width, C.height))
        
        for col_b in range(count_B_cols):
            for row_b in range(count_B_rows):
                x = leftover_left + used_w_A + col_b * B.width
                y = leftover_bottom + row_b * B.height
                if x + B.width <= box_w and y + B.height <= box_h:
                    layout.append((x, y, B, B.width, B.height))
        
        for col_d in range(count_B_cols):
            for row_d in range(count_C_rows):
                x = leftover_left + used_w_A + col_d * D.width
                y = leftover_bottom + used_h_A + row_d * D.height
                if x + D.width <= box_w and y + D.height <= box_h and not any(abs(px - x) < D.width and abs(py - y) < D.height for px, py, _, _, _ in layout):
                    layout.append((x, y, D, D.width, D.height))
        
        counts = {
            "A": count_A_cols * count_A_rows,
            "B": count_B_cols * count_B_rows,
            "C": count_C_cols * count_C_rows,
            "D": len([p for p in layout if p[2].name == "D"]),
        }
        
        total_power = (
            counts["A"] * A.power +
            counts["B"] * B.power +
            counts["C"] * C.power +
            counts["D"] * D.power
        )
    
    psu_count = 0
    psu_assignments = None
    if psu_wattage is not None:
        effective_wattage = psu_wattage * psu_max_load
        psu_count = math.ceil(total_power / effective_wattage) if total_power > 0 else 1
        psu_assignments = []
        
        if box_h >= box_w:
            sorted_layout = sorted(layout, key=lambda p: (p[1], p[0]))
            current_psu = 0
            current_power = 0
            current_row = None
            target_power_per_psu = total_power / psu_count if psu_count > 0 else total_power
            tolerance = target_power_per_psu * 0.1
            row_power = 0
            
            for x, y, panel, w, h in sorted_layout:
                power = panel.power * (w / panel.width) * (h / panel.height)
                if current_row is None or y >= current_row + h:
                    if current_row is not None:
                        if current_power + row_power > target_power_per_psu + tolerance and current_psu < psu_count - 1:
                            current_psu += 1
                            current_power = row_power
                        else:
                            current_power += row_power
                    current_row = y
                    row_power = power
                else:
                    row_power += power
                psu_assignments.append((x, y, panel, w, h, current_psu))
            if row_power > 0:
                current_power += row_power
        else:
            sorted_layout = sorted(layout, key=lambda p: (p[0], p[1]))
            current_psu = 0
            current_power = 0
            current_col = None
            target_power_per_psu = total_power / psu_count if psu_count > 0 else total_power
            tolerance = target_power_per_psu * 0.1
            col_power = 0
            
            for x, y, panel, w, h in sorted_layout:
                power = panel.power * (w / panel.width) * (h / panel.height)
                if current_col is None or x >= current_col + w:
                    if current_col is not None:
                        if current_power + col_power > target_power_per_psu + tolerance and current_psu < psu_count - 1:
                            current_psu += 1
                            current_power = col_power
                        else:
                            current_power += col_power
                    current_col = x
                    col_power = power
                else:
                    col_power += power
                psu_assignments.append((x, y, panel, w, h, current_psu))
            if col_power > 0:
                current_power += col_power
    
    return {
        "boards": counts,
        "leftover_left": leftover_left,
        "leftover_right": leftover_right,
        "leftover_top": leftover_top,
        "leftover_bottom": leftover_bottom,
        "total_power": total_power,
        "layout": layout,
        "orientation": orientation,
        "used_width": total_width_used,
        "used_height": total_height_used,
        "panels": panels,
        "psu_count": psu_count,
        "psu_assignments": psu_assignments,
        "psu_wattage": psu_wattage,
    }

def format_data_text(res, label, is_gridflex=False):
    c = res["boards"]
    leftover_left = res["leftover_left"]
    leftover_right = res["leftover_right"]
    leftover_top = res["leftover_top"]
    leftover_bottom = res["leftover_bottom"]
    psu_count = res["psu_count"]
    psu_assignments = res["psu_assignments"]
    psu_powers = [0] * psu_count if psu_assignments else []
    
    if psu_assignments:
        for _, _, panel, w, h, psu_idx in psu_assignments:
            psu_powers[psu_idx] += panel.power * (w / panel.width) * (h / panel.height)
    
    txt = f"{label:<15}\n"
    txt += f"{'Panels used:':<15}\n"
    if is_gridflex:
        panel_name = list(c.keys())[0]
        txt += f" {panel_name}: {c[panel_name]:<3}"
        if psu_count > 0:
            txt += f" PSU1: {math.ceil(psu_powers[0])}W"
        txt += "\n"
    else:
        txt += f" A: {c['A']:<3} B: {c['B']:<3}"
        if psu_count > 0:
            txt += f" PSU1: {math.ceil(psu_powers[0])}W"
        txt += "\n"
        txt += f" C: {c['C']:<3} D: {c['D']:<3}"
        if psu_count > 1:
            txt += f" PSU2: {math.ceil(psu_powers[1])}W"
        txt += "\n"
    
    txt += f"Total Power: {res['total_power']:.1f}W"
    if psu_count > 2:
        txt += f" PSU3: {math.ceil(psu_powers[2])}W"
    txt += "\n"
    
    for i in range(3, min(psu_count, 10)):
        txt += f"{'':<24}PSU{i+1}: {math.ceil(psu_powers[i])}W\n"
    
    txt += f"PSUs Needed: {psu_count}\n"
    if res["psu_wattage"] is not None:
        txt += f"PSU Wattage: {res['psu_wattage']}W\n"
    txt += f"Leftover margins (mm):\n"
    txt += f" Left: {leftover_left:.1f}\n"
    txt += f" Right: {leftover_right:.1f}\n"
    txt += f" Top: {leftover_top:.1f}\n"
    txt += f" Bottom: {leftover_bottom:.1f}"
    
    return txt

def generate_plot_data(box_w, box_h, layout, leftovers, orientation, psu_assignments, is_gridflex):
    colors = {
        "A": "rgba(76, 175, 80, 0.7)",
        "B": "rgba(129, 199, 132, 0.7)",
        "C": "rgba(174, 213, 129, 0.7)",
        "D": "rgba(197, 225, 165, 0.7)",
        "GridFlex": "rgba(0, 188, 212, 0.7)",
        "GridFlex DW": "rgba(233, 30, 99, 0.7)",
        "GridFlex RGBW": "rgba(255, 152, 0, 0.7)",
    }
    psu_colors = ["rgba(33, 150, 243, 0.7)", "rgba(244, 67, 54, 0.7)", "rgba(156, 39, 176, 0.7)", "rgba(255, 152, 0, 0.7)", "rgba(0, 188, 212, 0.7)"]
    
    shapes = []
    annotations = []
    # Outer rectangle
    shapes.append({
        "type": "rect",
        "x0": 0,
        "y0": 0,
        "x1": box_w,
        "y1": box_h,
        "line": {"color": "black", "width": 2},
        "fillcolor": "rgba(0,0,0,0)"
    })
    
    # Leftover margins
    if leftovers["leftover_left"] > 0:
        shapes.append({
            "type": "rect",
            "x0": 0,
            "y0": 0,
            "x1": leftovers["leftover_left"],
            "y1": box_h,
            "fillcolor": "rgba(224,224,224,0.4)",
            "line": {"width": 0}
        })
    if leftovers["leftover_right"] > 0:
        shapes.append({
            "type": "rect",
            "x0": box_w - leftovers["leftover_right"],
            "y0": 0,
            "x1": box_w,
            "y1": box_h,
            "fillcolor": "rgba(224,224,224,0.4)",
            "line": {"width": 0}
        })
    if leftovers["leftover_top"] > 0:
        shapes.append({
            "type": "rect",
            "x0": 0,
            "y0": box_h - leftovers["leftover_top"],
            "x1": box_w,
            "y1": box_h,
            "fillcolor": "rgba(224,224,224,0.4)",
            "line": {"width": 0}
        })
    if leftovers["leftover_bottom"] > 0:
        shapes.append({
            "type": "rect",
            "x0": 0,
            "y0": 0,
            "x1": box_w,
            "y1": leftovers["leftover_bottom"],
            "fillcolor": "rgba(224,224,224,0.4)",
            "line": {"width": 0}
        })
    
    # Panels
    for x, y, p, w, h, *extra in (psu_assignments if psu_assignments else [(x, y, p, w, h) for x, y, p, w, h in layout]):
        psu_idx = extra[0] if psu_assignments else None
        color = colors[p.name] if psu_idx is None else psu_colors[psu_idx % len(psu_colors)]
        if is_gridflex and (abs(w - p.width) > 0.001 or abs(h - p.height) > 0.001):
            color = f"rgba({int(float(color.split(',')[0].split('(')[1])*1.5)},{int(float(color.split(',')[1])*1.5)},{int(float(color.split(',')[2])*1.5)},0.7)"
        
        shapes.append({
            "type": "rect",
            "x0": x,
            "y0": y,
            "x1": x + w,
            "y1": y + h,
            "fillcolor": color,
            "line": {"color": "black", "width": 1.5}
        })
        
        if not is_gridflex:
            cx = x + w / 2
            cy = y + h / 2
            rotation = 90 if orientation == "Landscape" else 0
            font_size = max(6, min(w, h) * 0.15)
            annotations.append({
                "x": cx,
                "y": cy,
                "text": p.name,
                "showarrow": False,
                "font": {"size": font_size, "color": "black"},
                "textangle": rotation
            })
        
        if is_gridflex:
            num_width_cuts = int(round(w / p.cut_length)) if p.cut_length > 0 else 0
            for i in range(1, num_width_cuts):
                shapes.append({
                    "type": "line",
                    "x0": x + i * p.cut_length,
                    "y0": y,
                    "x1": x + i * p.cut_length,
                    "y1": y + h,
                    "line": {"color": "rgba(0,0,0,0.5)", "width": 1, "dash": "dash"}
                })
            num_height_cuts = int(round(h / p.cut_height)) if p.cut_height > 0 else 0
            if p.name == "GridFlex RGBW" and abs(h - 305) < 0.001:
                num_height_cuts = 6
                cut_height = 305 / 6
                for j in range(1, num_height_cuts):
                    shapes.append({
                        "type": "line",
                        "x0": x,
                        "y0": y + j * cut_height,
                        "x1": x + w,
                        "y1": y + j * cut_height,
                        "line": {"color": "rgba(0,0,0,0.5)", "width": 1, "dash": "dash"}
                    })
            else:
                for j in range(1, num_height_cuts):
                    shapes.append({
                        "type": "line",
                        "x0": x,
                        "y0": y + j * p.cut_height,
                        "x1": x + w,
                        "y1": y + j * p.cut_height,
                        "line": {"color": "rgba(0,0,0,0.5)", "width": 1, "dash": "dash"}
                    })
    
    layout_plot = {
        "shapes": shapes,
        "annotations": annotations,
        "xaxis": {"range": [0, box_w], "title": "Width (mm)"},
        "yaxis": {"range": [0, box_h], "title": "Height (mm)", "scaleanchor": "x", "scaleratio": 1},
        "title": f"Panel Layout ({orientation})",
        "showlegend": False,
        "margin": {"l": 50, "r": 50, "t": 50, "b": 50}
    }
    
    return layout_plot

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        box_w = float(request.form['box_w'])
        box_h = float(request.form['box_h'])
        if box_w <= 0 or box_h <= 0 or box_w < 52 or box_h < 60:
            raise ValueError("Dimensions too small")
        
        psu_wattage = None
        psu_max_load = 0.8
        psu_input = request.form['psu_wattage'].strip()
        if psu_input:
            psu_wattage = float(psu_input)
            if psu_wattage < 75:
                raise ValueError("PSU wattage must be at least 75 W")
        
        psu_load_input = request.form['psu_max_load'].strip()
        if psu_load_input:
            psu_max_load = float(psu_load_input)
            if psu_max_load < 1 or psu_max_load > 100:
                raise ValueError("PSU max load must be between 1 and 100")
            psu_max_load /= 100.0
        
        config = request.form['panel_type']
        is_gridflex = config in GRIDFLEX_PANELS
        if is_gridflex:
            panels = {config: GRIDFLEX_PANELS[config]}
        else:
            panels = PANELS.copy()
            if config == "GRID v4":
                panels["A"] = Panel("A", 312, 300, 6.12)
                panels["B"] = Panel("B", 312, 60, 1.13)
                panels["C"] = Panel("C", 52, 300, 1.3)
                panels["D"] = Panel("D", 52, 60, 0.32)
        
        # Portrait layout
        res1 = fit_layout(box_w, box_h, panels.copy(), "Portrait", psu_wattage, psu_max_load, is_gridflex)
        result_text1 = format_data_text(res1, "Layout 1", is_gridflex)
        plot_data1 = generate_plot_data(box_w, box_h, res1["layout"], {
            "leftover_left": res1["leftover_left"],
            "leftover_right": res1["leftover_right"],
            "leftover_top": res1["leftover_top"],
            "leftover_bottom": res1["leftover_bottom"]
        }, "Portrait", res1["psu_assignments"], is_gridflex)
        
        # Landscape layout (swapped dimensions)
        res2 = fit_layout(box_h, box_w, panels.copy(), "Portrait", psu_wattage, psu_max_load, is_gridflex)
        result_text2 = format_data_text(res2, "Layout 2", is_gridflex)
        plot_data2 = generate_plot_data(box_h, box_w, res2["layout"], {
            "leftover_left": res2["leftover_left"],
            "leftover_right": res2["leftover_right"],
            "leftover_top": res2["leftover_top"],
            "leftover_bottom": res2["leftover_bottom"]
        }, "Portrait", res2["psu_assignments"], is_gridflex)
        
        return jsonify({
            "layout1": {"text": result_text1, "plot": plot_data1},
            "layout2": {"text": result_text2, "plot": plot_data2}
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
