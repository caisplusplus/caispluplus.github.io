function initSphere() {
    let $canvas = $('#blob canvas'),
        canvas = $canvas[0],
        renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            context: canvas.getContext('webgl2'),
            antialias: true,
            alpha: true
        });

    renderer.setSize($canvas.width(), $canvas.height());
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setClearColor( 0x000000, 0 ); // background

    var controls;
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(75, $canvas.width() / $canvas.height(), 0.1, 1000);
    var innerColor = 0xff0000,
        outerColor = 0xff9900;
    var innerSize = 55,
        outerSize = 60;

    camera.position.z = -400;
// Mesh
    var group = new THREE.Group();
    scene.add(group);

// Lights
    var light = new THREE.AmbientLight( 0x404040 ); // soft white light
    scene.add( light );

    var directionalLight = new THREE.DirectionalLight( 0xffffff, 1 );
    directionalLight.position.set( 0, 128, 128 );
    scene.add( directionalLight );

// Sphere Wireframe Inner
    var sphereWireframeInner = new THREE.Mesh(
        new THREE.IcosahedronGeometry( innerSize, 2 ),
        new THREE.MeshPhongMaterial({
            color: innerColor,
            ambient: innerColor,
            wireframe: true,
            transparent: true,
            //alphaMap: THREE.ImageUtils.loadTexture( 'javascripts/alphamap.jpg' ),
            shininess: 100
        })
    );
    scene.add(sphereWireframeInner);

// Sphere Wireframe Outer
    var sphereWireframeOuter = new THREE.Mesh(
        new THREE.IcosahedronGeometry( outerSize, 3 ),
        new THREE.MeshLambertMaterial({
            color: outerColor,
            ambient: outerColor,
            wireframe: true,
            transparent: true,
            //alphaMap: THREE.ImageUtils.loadTexture( 'javascripts/alphamap.jpg' ),
            shininess: 0
        })
    );
    scene.add(sphereWireframeOuter);


// Sphere Glass Inner
    var sphereGlassInner = new THREE.Mesh(
        new THREE.SphereGeometry( innerSize, 32, 32 ),
        new THREE.MeshPhongMaterial({
            color: innerColor,
            ambient: innerColor,
            transparent: true,
            shininess: 25,
            //alphaMap: THREE.ImageUtils.loadTexture( 'javascripts/twirlalphamap.jpg' ),
            opacity: 0.3,
        })
    );
    scene.add(sphereGlassInner);

// Sphere Glass Outer
    var sphereGlassOuter = new THREE.Mesh(
        new THREE.SphereGeometry( outerSize, 32, 32 ),
        new THREE.MeshPhongMaterial({
            color: outerColor,
            ambient: outerColor,
            transparent: true,
            shininess: 25,
            //alphaMap: THREE.ImageUtils.loadTexture( 'javascripts/twirlalphamap.jpg' ),
            opacity: 0.3,
        })
    );
//scene.add(sphereGlassOuter);

// Particles Outer
    var geometry = new THREE.Geometry();
    for (i = 0; i < 35000; i++) {

        var x = -1 + Math.random() * 2;
        var y = -1 + Math.random() * 2;
        var z = -1 + Math.random() * 2;
        var d = 1 / Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2) + Math.pow(z, 2));
        x *= d;
        y *= d;
        z *= d;

        var vertex = new THREE.Vector3(
            x * outerSize,
            y * outerSize,
            z * outerSize
        );

        geometry.vertices.push(vertex);

    }


    var particlesOuter = new THREE.PointCloud(geometry, new THREE.PointCloudMaterial({
            size: 0.1,
            color: outerColor,
            //map: THREE.ImageUtils.loadTexture( 'javascripts/particletextureshaded.png' ),
            transparent: true,
        })
    );
    scene.add(particlesOuter);

// Particles Inner
    var geometry = new THREE.Geometry();
    for (i = 0; i < 35000; i++) {

        var x = -1 + Math.random() * 2;
        var y = -1 + Math.random() * 2;
        var z = -1 + Math.random() * 2;
        var d = 1 / Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2) + Math.pow(z, 2));
        x *= d;
        y *= d;
        z *= d;

        var vertex = new THREE.Vector3(
            x * outerSize,
            y * outerSize,
            z * outerSize
        );

        geometry.vertices.push(vertex);

    }


    var particlesInner = new THREE.PointCloud(geometry, new THREE.PointCloudMaterial({
            size: 0.1,
            color: innerColor,
            //map: THREE.ImageUtils.loadTexture( 'javascripts/particletextureshaded.png' ),
            transparent: true,
        })
    );
    scene.add(particlesInner);


    camera.position.z = -110;
//camera.position.x = mouseX * 0.05;
//camera.position.y = -mouseY * 0.05;
//camera.lookAt(scene.position);

    var time = new THREE.Clock();

    var render = function () {
        //camera.position.x = mouseX * 0.05;
        //camera.position.y = -mouseY * 0.05;
        camera.lookAt(scene.position);

        sphereWireframeInner.rotation.x += 0.002;
        sphereWireframeInner.rotation.z += 0.002;

        sphereWireframeOuter.rotation.x += 0.001;
        sphereWireframeOuter.rotation.z += 0.001;

        sphereGlassInner.rotation.y += 0.005;
        sphereGlassInner.rotation.z += 0.005;

        sphereGlassOuter.rotation.y += 0.01;
        sphereGlassOuter.rotation.z += 0.01;

        particlesOuter.rotation.y += 0.0005;
        particlesInner.rotation.y -= 0.002;

        var innerShift = Math.abs(Math.cos(( (time.getElapsedTime()+2.5) / 20)));
        var outerShift = Math.abs(Math.cos(( (time.getElapsedTime()+5) / 10)));

        sphereWireframeOuter.material.color.setHSL(0, 1, outerShift);
        sphereGlassOuter.material.color.setHSL(0, 1, outerShift);
        particlesOuter.material.color.setHSL(0, 1, outerShift);

        sphereWireframeInner.material.color.setHSL(0.08, 1, innerShift);
        particlesInner.material.color.setHSL(0.08, 1, innerShift);
        sphereGlassInner.material.color.setHSL(0.08, 1, innerShift);

        sphereWireframeInner.material.opacity = Math.abs(Math.cos((time.getElapsedTime()+0.5)/0.9)*0.5);
        sphereWireframeOuter.material.opacity = Math.abs(Math.cos(time.getElapsedTime()/0.9)*0.5);


        directionalLight.position.x = Math.cos(time.getElapsedTime()/0.5)*128;
        directionalLight.position.y = Math.cos(time.getElapsedTime()/0.5)*128;
        directionalLight.position.z = Math.sin(time.getElapsedTime()/0.5)*128;

        // controls.update();

        renderer.render(scene, camera);
        requestAnimationFrame(render);
    };

    render();
}

$(document).ready(function() {
    initSphere();
    setTimeout(function () {
        document.querySelector('#blob canvas').style.visibility = 'visible';
    }, 100);
});

$( window ).resize(function() {
    if (window.innerWidth < document.querySelector('canvas').width) {
        $('canvas').width(0.7 * window.innerWidth);
        initSphere();
    }
});