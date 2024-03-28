import { Group, Text, rem } from '@mantine/core'
import { IconUpload, IconFileUpload, IconX } from '@tabler/icons-react'
import { Dropzone, MIME_TYPES } from '@mantine/dropzone'
import { useEffect } from 'react'
import classes from './DragDrop.module.css'
import cx from 'clsx'

export interface DragDropProps {
  files: File[]
  setFiles: any
}

export const DragDrop = ({ files, setFiles }: DragDropProps) => {
  useEffect(() => {
    console.log(files)
  }, [files])

  return (
    <Dropzone
      className={cx(classes.dragdrop)}
      onDrop={(files: File[]) => {
        setFiles((prevFiles: File[]) => [...prevFiles, ...files])
      }}
      onReject={(files) => console.log('rejected files', files)}
      accept={[MIME_TYPES.mp4, MIME_TYPES.pdf]}
    >
      <Group
        className={cx(classes.dragdrop_content)}
        style={{ minHeight: rem(300), pointerEvents: 'none' }}
      >
        <Dropzone.Accept>
          <IconUpload
            size="3.2rem"
            stroke={1.5}
          />
        </Dropzone.Accept>
        <Dropzone.Reject>
          <IconX
            size="3.2rem"
            stroke={1.5}
          />
        </Dropzone.Reject>
        <Dropzone.Idle>
          <IconFileUpload
            size="3.2rem"
            stroke={1.5}
          />
        </Dropzone.Idle>

        <div>
          <Text
            size="xl"
            inline
          >
            Drag and drop here or click to select files
          </Text>
          <Text
            size="sm"
            color="dimmed"
            inline
            mt={7}
          >
            Attach as many lecture videos or supplementary pdfs as you like
          </Text>
        </div>
      </Group>
    </Dropzone>
  )
}
