import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import time
import uuid
import tempfile
import itertools
from typing import *
import atexit
import html
import json
from concurrent.futures import ThreadPoolExecutor
import shutil
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
import click


TEMP_DIR = Path(tempfile.gettempdir(), 'moge')
GRADIO_FILE_URL_PREFIX = '/gradio_api/file='
THREEJS_CDN = 'https://cdn.jsdelivr.net/npm/three@0.169.0'

# Client-side point cloud viewer. Instead of shipping a ~16 bytes/point GLB, the server sends the
# lossless float32 depth map (byte-plane shuffled so that HTTP gzip can actually compress it) plus
# the camera intrinsics, and the browser unprojects it into the exact same point cloud.
POINT_CLOUD_VIEWER_HTML = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #ffffff;
               font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  canvas { display: block; }
  #status { position: absolute; left: 0; right: 0; top: 50%; transform: translateY(-50%);
            text-align: center; color: #555; font-size: 14px; padding: 0 16px; }
  #hud { position: absolute; left: 8px; bottom: 8px; color: #666; font-size: 11px;
         background: rgba(255, 255, 255, 0.75); padding: 2px 6px; border-radius: 4px; }
</style>
<script type="importmap">
{"imports": {"three": "__CDN__/build/three.module.js", "three/addons/": "__CDN__/examples/jsm/"}}
</script>
</head>
<body>
<div id="status">Loading&hellip;</div>
<div id="hud"></div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const CONFIG = /*__CONFIG__*/null;

const statusEl = document.getElementById('status');
const hudEl = document.getElementById('hud');
const setStatus = (t) => { statusEl.textContent = t; statusEl.style.display = t ? 'block' : 'none'; };

async function fetchDepth(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error('depth: HTTP ' + res.status);
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        setStatus('Loading\u2026 ' + (received / 1048576).toFixed(1) + ' MB');
    }
    const buf = new Uint8Array(received);
    let offset = 0;
    for (const c of chunks) { buf.set(c, offset); offset += c.length; }
    // Undo the byte-plane shuffle: [b0 x N][b1 x N][b2 x N][b3 x N] -> little-endian float32.
    const n = received >> 2;
    const bytes = new Uint8Array(received);
    for (let k = 0; k < 4; k++) {
        const off = k * n;
        for (let i = 0; i < n; i++) bytes[(i << 2) | k] = buf[off + i];
    }
    return new Float32Array(bytes.buffer);
}

async function fetchColor(url, width, height) {
    const res = await fetch(url);
    if (!res.ok) throw new Error('color: HTTP ' + res.status);
    const bitmap = await createImageBitmap(await res.blob(),
        { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    return ctx.getImageData(0, 0, width, height).data;
}

function buildGeometry(depth, rgba) {
    const { width: W, height: H, fx, fy, cx, cy } = CONFIG;
    let count = 0;
    for (let i = 0; i < W * H; i++) if (depth[i] > 0) count++;
    const positions = new Float32Array(count * 3);
    const colors = new Uint8Array(count * 3);
    const depths = new Float64Array(count);
    let j = 0, i = 0;
    for (let y = 0; y < H; y++) {
        const v = (y + 0.5) / H;
        for (let x = 0; x < W; x++, i++) {
            const d = depth[i];
            if (!(d > 0)) continue;
            // OpenCV camera convention -> three.js (x right, y up, z backward)
            positions[j * 3] = ((x + 0.5) / W - cx) / fx * d;
            positions[j * 3 + 1] = -((v - cy) / fy * d);
            positions[j * 3 + 2] = -d;
            colors[j * 3] = rgba[i * 4];
            colors[j * 3 + 1] = rgba[i * 4 + 1];
            colors[j * 3 + 2] = rgba[i * 4 + 2];
            depths[j] = d;
            j++;
        }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(colors, 3, true));
    const sorted = Array.from(depths).sort((a, b) => a - b);
    const median = sorted.length ? sorted[sorted.length >> 1] : 1.0;
    return { geometry, count, median };
}

async function main() {
    const { width: W, height: H, fx, fy } = CONFIG;
    const [depth, rgba] = await Promise.all([
        fetchDepth(CONFIG.depthUrl),
        fetchColor(CONFIG.colorUrl, W, H),
    ]);
    setStatus('Building point cloud\u2026');
    await new Promise(requestAnimationFrame);
    const { geometry, count, median } = buildGeometry(depth, rgba);

    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xffffff, 1);
    document.body.appendChild(renderer.domElement);

    const material = new THREE.ShaderMaterial({
        uniforms: {
            uSizeFactor: { value: CONFIG.pointScale / (fx * W) },
            uHalfHeight: { value: 1.0 },
        },
        vertexShader: [
            'uniform float uSizeFactor;',
            'uniform float uHalfHeight;',
            'attribute vec3 aColor;',
            'varying vec3 vColor;',
            'void main() {',
            '  vColor = aColor;',
            '  vec4 mv = modelViewMatrix * vec4(position, 1.0);',
            '  gl_Position = projectionMatrix * mv;',
            '  float worldSize = uSizeFactor * max(-position.z, 1e-4);',
            '  float px = worldSize * projectionMatrix[1][1] * uHalfHeight / max(-mv.z, 1e-4);',
            '  gl_PointSize = clamp(px, 1.0, 24.0);',
            '}',
        ].join('\n'),
        fragmentShader: [
            'varying vec3 vColor;',
            'void main() { gl_FragColor = vec4(vColor, 1.0); }',
        ].join('\n'),
    });

    const scene = new THREE.Scene();
    scene.add(new THREE.Points(geometry, material));

    const camera = new THREE.PerspectiveCamera(50, 1, median * 1e-3, median * 1e3);
    camera.position.set(0, 0, 0);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, -median);
    controls.enableDamping = true;
    controls.dampingFactor = 0.15;

    function resize() {
        const w = Math.max(window.innerWidth, 1), h = Math.max(window.innerHeight, 1);
        renderer.setSize(w, h);
        camera.aspect = w / h;
        // Match the input image framing: widen the vertical fov when the container is narrower.
        const tanHalfY = 0.5 / fy;
        const tanHalfX = 0.5 / fx;
        camera.fov = 2 * Math.atan(Math.max(tanHalfY, tanHalfX / camera.aspect)) * 180 / Math.PI;
        camera.updateProjectionMatrix();
        material.uniforms.uHalfHeight.value = renderer.domElement.height * 0.5;
    }
    window.addEventListener('resize', resize);
    resize();

    setStatus('');
    hudEl.textContent = count.toLocaleString() + ' points \u00b7 drag to rotate, scroll to zoom';
    renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
}

main().catch((e) => { setStatus('Failed to load the 3D view: ' + e.message); });
</script>
</body>
</html>
'''


def build_viewer_html(config: Dict[str, Any], height: str = '60vh') -> str:
    document = POINT_CLOUD_VIEWER_HTML.replace('__CDN__', THREEJS_CDN).replace('/*__CONFIG__*/null', json.dumps(config))
    return (
        f'<iframe srcdoc="{html.escape(document, quote=True)}" '
        f'style="display:block;box-sizing:border-box;width:100%;height:{height};'
        f'border:1px solid var(--block-border-color,#e5e5e5);border-radius:8px;background:#fff"></iframe>'
    )


@click.command(help='Web demo')
@click.option('--share', is_flag=True, help='Whether to run the app in shared mode.')
@click.option('--pretrained', 'pretrained_model_name_or_path', required=True, help='The name or path of the pre-trained model.')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Whether to use fp16 inference.')
def main(share: bool, pretrained_model_name_or_path: str, use_fp16: bool):
    print("Import modules...")
    # Lazy import
    import cv2
    import torch
    import numpy as np
    import trimesh
    import trimesh.visual
    from PIL import Image
    import gradio as gr
    try:
        import spaces   # This is for deployment at huggingface.co/spaces
        HUGGINFACE_SPACES_INSTALLED = True
    except ImportError:
        HUGGINFACE_SPACES_INSTALLED = False

    import flex_gemm
    flex_gemm.config.AUTOTUNE_MODE = 'never'

    try:
        import utils3d_moge as utils3d
    except ImportError:
        import utils3d
    from moge.utils.io import write_normal
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.model import import_model_class_by_version
    from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
    from moge.utils.tools import timeit

    print("Load model...")
    # v3 not released to huggingface yet
    # if pretrained_model_name_or_path is None:
    #     DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
    #         "v1": "Ruicheng/moge-vitl",
    #         "v2": "Ruicheng/moge-2-vitl-normal",
    #     }
    #     pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION['v3']
    model = import_model_class_by_version('v3').from_pretrained(pretrained_model_name_or_path).cuda().eval()
    thread_pool_executor = ThreadPoolExecutor(max_workers=1)
    TEMP_DIR.mkdir(exist_ok=True)

    def delete_later(path: Union[str, os.PathLike], delay: int = 300):
        def _delete():
            try: 
                os.remove(path) 
            except FileNotFoundError:
                pass
        def _wait_and_delete():
            time.sleep(delay)
            _delete()
        thread_pool_executor.submit(_wait_and_delete)
        atexit.register(_delete)

    # Inference on GPU. 
    @(spaces.GPU if HUGGINFACE_SPACES_INSTALLED else lambda x: x)
    def run_with_gpu(image: np.ndarray, resolution_level: int, apply_mask: bool, refine_steps: int) -> Dict[str, np.ndarray]:
        image_tensor = torch.tensor(image, dtype=torch.float32, device=torch.device('cuda')).permute(2, 0, 1) / 255
        output = model.infer(image_tensor, apply_mask=apply_mask, resolution_level=resolution_level, refine_steps=refine_steps, use_fp16=use_fp16)
        output = {k: v.cpu().numpy() for k, v in output.items() if isinstance(v, torch.Tensor)}
        return output

    # Full inference pipeline
    def run(image: np.ndarray, max_size: int = 800, resolution_level: str = 'High',   apply_mask: bool = True, remove_edge: bool = True, refine_steps: int = 3, request: gr.Request = None):
        larger_size = max(image.shape[:2])
        if larger_size > max_size:
            scale = max_size / larger_size
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        height, width = image.shape[:2]

        resolution_level_int = {'Low': 0, 'Medium': 5, 'High': 9, 'Ultra': 30}.get(resolution_level, 9)
        output = run_with_gpu(image, resolution_level_int, apply_mask, refine_steps)

        points, depth, mask, normal = output['points'], output['depth'], output['mask'], output['normal']

        if remove_edge:
            edge = utils3d.np.depth_map_edge(depth, ltol=0.02)
            mask_cleaned = mask & ~edge
        else:
            mask_cleaned = mask
        
        results = {
            **output,
            'mask_cleaned': mask_cleaned,
            'image': image
        }

        # depth & normal visualization
        depth_vis = colorize_depth(depth)
        normal_vis = colorize_normal(normal)
        mask_vis = mask_cleaned.astype(np.uint8) * 255

        # mesh & pointcloud
        faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
            points,
            image.astype(np.float32) / 255,
            utils3d.np.uv_map((height, width)),
            mask=mask_cleaned,
            tri=True
        )
        vertices = vertices * np.array([1, -1, -1], dtype=np.float32) 
        vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)

        TEMP_DIR.mkdir(exist_ok=True)
        output_path = Path(TEMP_DIR, request.session_hash)
        shutil.rmtree(output_path, ignore_errors=True)
        output_path.mkdir(exist_ok=True, parents=True)
        trimesh.Trimesh(
            vertices=vertices * np.array([-1, 1, -1], dtype=np.float32),
            faces=faces, 
            visual = trimesh.visual.texture.TextureVisuals(
                uv=vertex_uvs, 
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.fromarray(image),
                    metallicFactor=0.5,
                    roughnessFactor=1.0
                )
            ),
            process=False
        ).export(output_path / 'mesh.glb')
        trimesh.Trimesh(
            vertices=vertices, 
            faces=faces, 
            vertex_colors=vertex_colors,
            process=False
        ).export(output_path / 'mesh.ply')
        trimesh.PointCloud(
            vertices=vertices, 
            colors=vertex_colors,
        ).export(output_path / 'pointcloud.glb')
        trimesh.PointCloud(
            vertices=vertices, 
            colors=vertex_colors,
        ).export(output_path / 'pointcloud.ply')
        cv2.imwrite(str(output_path / 'depth.exr'), depth.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
        cv2.imwrite(str(output_path / 'points.exr'), cv2.cvtColor(points.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
        files = ['mesh.glb', 'mesh.ply', 'pointcloud.glb', 'pointcloud.ply', 'depth.exr', 'points.exr']
        if normal is not None:
            cv2.imwrite(str(output_path / 'normal.exr'), cv2.cvtColor(normal.astype(np.float32) * np.array([1, -1, -1], dtype=np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            files.append('normal.exr')

        for f in files:
            delete_later(output_path / f)

        # FOV
        intrinsics = results['intrinsics']
        fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
        fov_x, fov_y = np.rad2deg([fov_x, fov_y])

        # Viewer payload: lossless depth + intrinsics (~5x smaller than the equivalent point cloud GLB).
        # Byte-plane shuffling groups the float32 exponents together so that HTTP gzip can compress them.
        depth_view = np.where(mask_cleaned & np.isfinite(depth), depth, 0).astype(np.float32)
        (output_path / 'view_depth.bin').write_bytes(
            np.ascontiguousarray(depth_view.reshape(-1).view(np.uint8).reshape(-1, 4).T).tobytes()
        )
        cv2.imwrite(str(output_path / 'view_color.webp'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 101])
        for f in ['view_depth.bin', 'view_color.webp']:
            delete_later(output_path / f)
        cache_key = uuid.uuid4().hex
        viewer_html_value = build_viewer_html({
            'width': width,
            'height': height,
            'fx': float(intrinsics[0, 0]),
            'fy': float(intrinsics[1, 1]),
            'cx': float(intrinsics[0, 2]),
            'cy': float(intrinsics[1, 2]),
            'pointScale': 1.4,
            'depthUrl': f'{GRADIO_FILE_URL_PREFIX}{(output_path / "view_depth.bin").as_posix()}?v={cache_key}',
            'colorUrl': f'{GRADIO_FILE_URL_PREFIX}{(output_path / "view_color.webp").as_posix()}?v={cache_key}',
        })

        # messages
        viewer_message = f'**Note:** Inference has been completed. The point cloud is streamed as a depth map and unprojected in your browser.'
        if resolution_level != 'Ultra':
            depth_message = f'**Note:** Want sharper depth map? Try increasing the `maximum image size` and setting the `inference resolution level` to `Ultra` in the settings.'
        else:
            depth_message = ""

        return (
            results,
            depth_vis,
            normal_vis, 
            mask_vis,
            viewer_html_value,
            [(output_path / f).as_posix() for f in files],
            f'**Horizontal FOV: {fov_x:.1f}°. Vertical FOV: {fov_y:.1f}°**',
            viewer_message,
            depth_message
        )

    def reset_measure(results: Dict[str, np.ndarray]):
        return [results['image'], [], ""]


    def measure(results: Dict[str, np.ndarray], measure_points: List[Tuple[int, int]], event: gr.SelectData):
        point2d = event.index[0], event.index[1]
        measure_points.append(point2d)

        image = results['image'].copy()
        for p in measure_points:
            image = cv2.circle(image, p, radius=5, color=(255, 0, 0), thickness=2)

        depth_text = ""
        for i, p in enumerate(measure_points):
            d = results['depth'][p[1], p[0]]
            depth_text += f"**P{i + 1} depth: {d:.2f}m.** "

        if len(measure_points) == 2:
            point1, point2 = measure_points
            image = cv2.line(image, point1, point2, color=(255, 0, 0), thickness=2)
            distance = np.linalg.norm(results['points'][point1[1], point1[0]] - results['points'][point2[1], point2[0]])
            measure_points = []

            distance_text = f"**Distance: {distance:.2f}m**"

            text = depth_text + distance_text
            return [image, measure_points, text]
        else:
            return [image, measure_points, depth_text]
        
    print("Create Gradio app...")
    with gr.Blocks() as demo:
        gr.Markdown(
            f"## Turn a 2D image into a 3D point map with [MoGe3](https://qft-333.github.io/moge3page/)\n Model: {pretrained_model_name_or_path}"
        )
        results = gr.State(value=None)
        measure_points = gr.State(value=[])

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", image_mode="RGB", label="Input Image")
                with gr.Accordion(label="Settings", open=False):
                    max_size_input = gr.Number(value=1600, label="Maximum Image Size", precision=0, minimum=256, maximum=4096)
                    refine_steps = gr.Number(value=3, label="Refine Steps", precision=0, minimum=0, maximum=5)
                    resolution_level = gr.Dropdown(['Low', 'Medium', 'High', 'Ultra'], label="Inference Resolution Level", value='High')
                    apply_mask = gr.Checkbox(value=True, label="Apply mask")
                    remove_edges = gr.Checkbox(value=True, label="Remove edges")
                submit_btn = gr.Button("Submit")

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("3D View"):
                        viewer_message = gr.Markdown("")
                        point_cloud_viewer = gr.HTML("", label="3D Point Map")
                        fov = gr.Markdown()
                    with gr.Tab("Depth"):
                        depth_message = gr.Markdown("")
                        depth_map = gr.Image(type="numpy", label="Colorized Depth Map", format='png', interactive=False)
                    with gr.Tab("Normal"):
                        normal_map = gr.Image(type="numpy", label="Normal Map", format='png', interactive=False)
                    with gr.Tab("Mask"):
                        mask_map = gr.Image(type="numpy", label="Mask", format='png', interactive=False)
                    with gr.Tab("Measure"):
                        gr.Markdown("### Click on the image to measure the distance between two points. \n"
                         "**Note:** Metric scale is most reliable for typical indoor or street scenes, and may degrade for contents unfamiliar to the model (e.g., stylized or close-up images).")
                        measure_image = gr.Image(type="numpy", show_label=False, format='webp', interactive=False, sources=[])
                        gr.Markdown("Click on the image to measure the distance between two points.")
                        measure_text = gr.Markdown("")
                    with gr.Tab("Download"):
                        files = gr.File(type='filepath', label="Output Files")

        if Path('example_images/moge3').exists():
            example_image_paths = sorted(list(itertools.chain(*[Path('example_images/moge3').glob(f'*.{ext}') for ext in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']])))
            examples = gr.Examples(
                examples = example_image_paths,
                inputs=input_image,
                label="Examples"
            )

        submit_btn.click(
            fn=lambda: [None, None, None, None, "", None, "", "", ""],
            outputs=[results, depth_map, normal_map, mask_map, point_cloud_viewer, files, fov, viewer_message, depth_message]
        ).then(
            fn=run,
            inputs=[input_image, max_size_input, resolution_level, apply_mask, remove_edges, refine_steps],
            outputs=[results, depth_map, normal_map, mask_map, point_cloud_viewer, files, fov, viewer_message, depth_message]
        ).then(
            fn=reset_measure,
            inputs=[results],
            outputs=[measure_image, measure_points, measure_text]
        )

        measure_image.select(
            fn=measure,
            inputs=[results, measure_points],
            outputs=[measure_image, measure_points, measure_text]
        )
    
    demo.launch(
        share=share,
        allowed_paths=[str(TEMP_DIR)],
        app_kwargs={
            "middleware": [
                Middleware(
                    GZipMiddleware,
                    minimum_size=1024,
                    compresslevel=6,
                )
            ]
        },
    )


if __name__ == '__main__':
    main()
