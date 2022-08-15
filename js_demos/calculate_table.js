let alpha = 1;
    if (sliceThickness < 1.5) {
      alpha = sliceThickness / 1.5;
    } else if (sliceThickness >= 3.5) {
      alpha = sliceThickness / 3.5;
    }

    const { lesions = {}, summary: { spacing: [x, y, z], score } = {} } =
    activeState;
    const result = Object.entries(lesions).reduce(
    (tmp, [k, v]) => {
      const {contour: { data },} = v;
      const index = tmp.findIndex((item) => item.type === k);
      if (index !== -1) {
        const ctaVolumePixel = data.reduce(
          (sum, d) => (sum += d.pixelArea),
          0
        );
        const ctaVolume = totofixed(ctaVolumePixel * x * y * z, 2);

        const equivalentMassHU = data.reduce((hu, d) => {
          if (d.avgHU !== -9999) {
            hu += d.pixelArea * d.avgHU;
          }
          return hu;
        }, 0);
        const equivalentMass = totofixed(
          (equivalentMassHU * cparameter * x * y * z) / 1000,
          2
        );

        let calcificationScore = data.reduce((calcScore, d) => {
          calcScore += d.agatstonPixelArea.reduce(
            (tScore, count, i) => (tScore += count * score[i]),
            0
          );
          return calcScore;
        }, 0);
        calcificationScore = totofixed(calcificationScore * alpha, 2);

        tmp[index] = {
          type: k,
          ctaVolume,
          equivalentMass,
          calcificationScore,
        };
        const total = tmp[tmp.length - 1];
        total.ctaVolume += Number(totofixed(ctaVolume, 2));
        total.equivalentMass += Number(totofixed(equivalentMass, 2));
        total.calcificationScore += Number(totofixed(calcificationScore, 2));
        
        total.ctaVolume = Number(totofixed(total.ctaVolume, 2));
        total.equivalentMass = Number(totofixed(total.equivalentMass, 2));
        total.calcificationScore = Number(
          totofixed(total.calcificationScore, 2)
        );
      }
      return tmp;
    },
    [
      {
        type: 'vessel1',
        ctaVolume: 0,
        equivalentMass: 0,
        calcificationScore: 0,
      },
      {
        type: 'vessel2',
        ctaVolume: 0,
        equivalentMass: 0,
        calcificationScore: 0,
      },
      {
        type: 'vessel5',
        ctaVolume: 0,
        equivalentMass: 0,
        calcificationScore: 0,
      },
      {
        type: 'vessel9',
        ctaVolume: 0,
        equivalentMass: 0,
        calcificationScore: 0,
      },
      {
        type: 'Total',
        ctaVolume: 0,
        equivalentMass: 0,
        calcificationScore: 0,
      },
    ]
  );
  return result;
}